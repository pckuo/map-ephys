# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os.path

import datajoint as dj
from pipeline import ephys

# %%


def _fetch_manual_label(metrics=['unit_amp', 'unit_snr', 'isi_violation', 'avg_firing_rate',
                                 'presence_ratio', 'amplitude_cutoff',
                                 'isolation_distance', 'l_ratio', 'd_prime', 'nn_hit_rate', 'nn_miss_rate',
                                 'max_drift',
                                 'cumulative_drift',
                                 'drift_metric',
                                 ]):
    """
    Fetch manual spike sorting data

    Parameters
    ----------
    metrics : list, optional
        by default ['unit_amp', 'unit_snr', 'isi_violation', 'avg_firing_rate', 'presence_ratio', 'amplitude_cutoff', 'isolation_distance', 'l_ratio', 'd_prime', 'nn_hit_rate', 'nn_miss_rate', 'max_drift', 'cumulative_drift', ]

    Returns
    -------
    X_scale, Y, scaler
    """
    # == Get queries ==
    sorters = (dj.U('note_source') & ephys.UnitNote).fetch('note_source')

    if os.path.isfile('./spike_sorting.pkl'):
        df_all = pd.read_pickle('./spike_sorting.pkl')
        print('loaded from pickle!')
    else:
        all_unit = ((ephys.Unit.proj('unit_amp', 'unit_snr')
                    & (ephys.ProbeInsertion & ephys.UnitNote).proj())  # All units from all sorted sessions
                    * ephys.UnitStat * ephys.ClusterMetric
                    * ephys.MAPClusterMetric.DriftMetric
                    * ephys.ProbeInsertion.RecordableBrainRegion()
                    ).proj(..., _='unit_quality')

        session_sorter = (dj.U('subject_id', 'session',
                               'insertion_number', 'note_source') & ephys.UnitNote)

        # == Get metrics ==
        print('fetching...', end='')
        df_all = all_unit.proj(*metrics,
                               ).fetch(format='frame').astype('float')

        # Reorgnize data, such that None = not sorted by this sorter; 0 = sorted but bad units; 1 = good units
        for sorter in sorters:

            # Set all units in sessions that have been sorted by this sorter to 1, otherwise None
            this_sorter = all_unit.aggr(session_sorter & f'note_source="{sorter}"',
                                        **{f'sess_{sorter}': f'sum(note_source="{sorter}")'},
                                        keep_all_rows=True)

            # Add sort note (1 is good or ok, but 0 and None is still ambiguous)
            this_sorter *= this_sorter.aggr(ephys.UnitNote & f'note_source="{sorter}"',
                                            **{sorter: f'count(note_source="{sorter}")'},
                                            keep_all_rows=True)

            # Finally, in sessions that have been sorted by this sorter, set all "good" or "ok" to 1, otherwise 0
            this_sorter = this_sorter.proj(..., f'-{sorter}',
                                           **{sorter: f'{sorter} & sess_{sorter}'})

            df_all[sorter] = this_sorter.fetch(sorter).astype('float')

        df_all.to_pickle('./spike_sorting.pkl')
        print('done!')

    # == Data cleansing ==
    # Deal with missing values (only keep units with all non-nan metrics)
    np.sum(np.all(~np.isnan(df_all), axis=1))
    isnan = np.sum(np.isnan(df_all), axis=0)
    metrics_allnotnan = isnan[isnan == 0].index

    # Training data set
    df = df_all.reset_index()[list(metrics_allnotnan) +
                              list(sorters) + ['brain_area']]

    # Summary of all labels
    # print(df.groupby(list(sorters), dropna=False).size().reset_index(name='Count'))

    # Flatten df such that the same unit could exist more than one times but with 0 or 1 labelling
    df_flatten = pd.melt(df, id_vars=list(metrics_allnotnan) + ['brain_area'],
                         value_vars=sorters, var_name='sorter', value_name='label')
    df_flatten = df_flatten[~np.isnan(df_flatten.label)].reset_index()

    # Normalize features
    X = df_flatten[metrics_allnotnan]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    # Add back columns name
    X_scaled = pd.DataFrame(data=X_scaled, columns=metrics_allnotnan)
    Y = df_flatten.iloc[:, -1].values

    return X, Y, X_scaled, scaler, df_flatten

# %%


def plot_hist_roc(score, Y, range=[0, 1], score_label=''):
    # Histogram and ROC
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].hist(score[Y == 0], 50, density=True, label='noise',
                color='k', alpha=0.7, range=range)
    axs[0].hist(score[Y == 1], 50, density=True, label='SU',
                color='g', alpha=0.7, range=range)
    axs[0].set(xlabel=score_label)
    axs[0].legend()

    fpr, tpr, thresholds = roc_curve(Y, score)
    axs[1].plot(fpr, tpr, 'g', lw=3,
                label=f'AUC = {roc_auc_score(Y, score):.4f}')
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set(xlabel='False positive rate', ylabel='True positive rate')
    axs[1].legend()


# %%
def do_clf(X, Y, clf_name='Linear SVC'):

    C = 10
    # kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

    classifiers = {
        'L1 logistic': LogisticRegression(C=C, penalty='l1',
                                          solver='saga',
                                          multi_class='multinomial',
                                          max_iter=10000,
                                          verbose=False),
        'L2 logistic (Multinomial)': LogisticRegression(C=C, penalty='l2',
                                                        solver='saga',
                                                        multi_class='multinomial',
                                                        max_iter=10000,
                                                        verbose=False),
        'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2',
                                                solver='saga',
                                                multi_class='ovr',
                                                max_iter=10000,
                                                verbose=False),
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,
                          random_state=0, verbose=False),

        'MLP': MLPClassifier(hidden_layer_sizes=[20],
                             activation='relu',
                             #  solver='sgd',
                             early_stopping=False,
                             max_iter=5000,
                             verbose=0)
        # 'GPC': GaussianProcessClassifier(kernel)
    }

    clf = classifiers[clf_name]
    scores = cross_val_score(clf, X, Y, cv=10)
    print(scores)
    print("%0.3f accuracy with a standard deviation of %0.3f" %
          (scores.mean(), scores.std()))

    clf.fit(X, Y)

    print('all score = ', clf.score(X, Y))

    # predicted = cross_val_predict(svc, X_scaled, Y, cv=10)
    # Plot data on the plane with largest coef_
    if clf_name == 'MLP':
        svc_coef = clf.coefs_
        plot_hist_roc(clf.predict_proba(X)[:, 1], Y,
                      score_label='SU probability (SVM)', range=[0, 1])
    else:
        svc_coef = clf.coef_
        plot_hist_roc(clf.decision_function(X), Y,
                      score_label='Decision function', range=[-20, 5])
        plot_hist_roc(clf.predict_proba(X)[:, 1], Y,
                      score_label='SU probability (SVM)', range=[0, 1])

    return svc_coef

# %%


def do_all_model_2d_probability(X, y):
    """
    Inspired by
    https://scikit-learn.org/stable/auto_examples/classification/plot_classification_probability.html
    """

    C = 10
    # kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

    classifiers = {
        'L1 logistic': LogisticRegression(C=C, penalty='l1',
                                          solver='saga',
                                          multi_class='multinomial',
                                          max_iter=10000),
        'L2 logistic (Multinomial)': LogisticRegression(C=C, penalty='l2',
                                                        solver='saga',
                                                        multi_class='multinomial',
                                                        max_iter=10000),
        'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2',
                                                solver='saga',
                                                multi_class='ovr',
                                                max_iter=10000),
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,
                          random_state=0),
        # 'GPC': GaussianProcessClassifier(kernel)
    }

    n_classifiers = len(classifiers)

    plt.figure(figsize=(3 * 2, n_classifiers * 2))
    plt.subplots_adjust(bottom=.2, top=.95)

    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X, y)

        y_pred = classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

        # Find the most two relavent dimensions to visualize
        max_idx = abs(classifier.coef_).argsort()[0][-1:-3:-1]
        xxx = X.iloc[:, max_idx[0]]
        yyy = X.iloc[:, max_idx[1]]

        xx = np.linspace(min(xxx), max(xxx), 100)
        yy = np.linspace(min(yyy), max(yyy), 100).T
        xx, yy = np.meshgrid(xx, yy)
        Xfull = np.zeros([xx.size, X.shape[1]])
        Xfull[:, max_idx[0]] = xx.ravel()
        Xfull[:, max_idx[1]] = yy.ravel()

        # View probabilities:
        probas = classifier.predict_proba(Xfull)
        n_classes = np.unique(y_pred).size
        for k in range(n_classes):
            plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
            plt.title("Class %d" % k)
            if k == 0:
                plt.ylabel(name)
            imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                       extent=(min(xxx), max(xxx), min(yyy), max(yyy)), origin='lower')
            plt.xticks(())
            plt.yticks(())
            idx = (y_pred == k)
            if idx.any():
                plt.scatter(X.iloc[idx, max_idx[0]], X.iloc[idx, max_idx[1]],
                            marker='o', c='w', edgecolor='k')
            if k == 0:
                plt.gca().set(
                    xlabel=X.columns[max_idx[0]], ylabel=X.columns[max_idx[1]])

            plt.gca().set(xlim=(-2, 2), ylim=(-2, 2))

    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

    plt.show()

# %%


def _plot_tsne(X_tsne, X, colors, color_name=[]):
    # == Plotting ==
    fig, axs = plt.subplots(1, 4, figsize=(27, 6))

    # Presence_ratio
    scatter = axs[0].scatter(X_tsne[:, 0], X_tsne[:, 1],
                             c=colors, s=(X['presence_ratio'])**2*50, edgecolors='none', alpha=.3)
    handles, labels = scatter.legend_elements()
    axs[0].add_artist(axs[0].legend(handles, color_name if len(
        color_name) else labels, loc="lower right", title=""))

    handles, labels = scatter.legend_elements(num=7,
                                              prop="sizes", alpha=0.6, func=lambda x: np.sqrt(x / 50))
    axs[0].legend(handles, labels, loc="upper right", title="presence_ratio")

    # Amp cutoff
    scatter = axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                             c=colors, s=(1 - X['amplitude_cutoff'])**2*50, edgecolors='none', alpha=.3)
    handles, labels = scatter.legend_elements(num=7,
                                              prop="sizes", alpha=0.6, func=lambda x: 1 - np.sqrt(x / 50))
    axs[1].legend(handles, labels, loc="upper right", title="amp_cutoff")

    # # Unit amp
    # axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
    #                c=c, s=(X['unit_amp'])**2*0.001, edgecolors='none', alpha=.3)
    # handles, labels = scatter.legend_elements(num=7,
    #                                           prop="sizes", alpha=0.6, func=lambda x: np.sqrt(x / 0.001))
    # axs[1].legend(handles, labels, loc="upper right", title="unit_amp")

    # isi_violation
    axs[2].scatter(X_tsne[:, 0], X_tsne[:, 1],
                   c=colors, s=(np.maximum(1, 10-X['isi_violation'])) * 5, edgecolors='none', alpha=.3)
    handles, labels = scatter.legend_elements(num=7,
                                              prop="sizes", alpha=0.6, func=lambda x: 10 - x / 5)
    axs[2].legend(handles, labels, loc="upper right", title="isi_violation")

    # drift_metric
    axs[3].scatter(X_tsne[:, 0], X_tsne[:, 1],
                   c=colors, s=(1.5 - X['drift_metric'])**2*50, edgecolors='none', alpha=.3)
    handles, labels = scatter.legend_elements(num=7,
                                              prop="sizes", alpha=0.6, func=lambda x: 1 - np.sqrt(x / 50))
    axs[3].legend(handles, labels, loc="upper right", title="drift_metric")


# %%
def do_tsne(X_scaled, Y, X, areas):
    # TSNE
    # == TSNE ==
    tsne = TSNE(n_components=2,
                perplexity=30,
                learning_rate=200,
                random_state=0,
                )
    X_tsne = tsne.fit_transform(X_scaled)

    _plot_tsne(X_tsne, X, ['green' if y else 'gray' for y in Y])
    _plot_tsne(X_tsne, X, [np.where(u == areas.unique())[0][0]
               for u in areas], color_name=areas.unique())


# %%
# Run it
X, Y, X_scaled, scaler, df_flatten = _fetch_manual_label()
areas = df_flatten.brain_area

svc_coef = do_clf(X_scaled[areas == 'Striatum'],
                  Y[areas == 'Striatum'],
                  clf_name='MLP')

# %%
print(np.vstack([X.columns, svc_coef]).T)

# %% Tsne
aoi = areas != 'ALefefM'
do_tsne(X_scaled[aoi],
        Y[aoi],
        X[aoi], 
        areas[aoi])

# %%
do_all_model_2d_probability(X_scaled, Y)
