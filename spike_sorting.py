# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import check_scoring
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import os.path

import datajoint as dj
from pipeline import ephys


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

    if os.path.isfile('./export/spike_sorting.pkl'):
        df_all = pd.read_pickle('./export/spike_sorting.pkl')
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

        df_all.to_pickle('./export/spike_sorting.pkl')
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


def false_neg_and_pos(clf, x, y):
    y_pred = clf.predict(x)
    false_pos = sum((y_pred == 1) & (y == 0))
    false_neg = sum((y_pred == 0) & (y == 1))
    all_pos = sum((y_pred == 1) | (y == 1))
    return false_neg/all_pos, false_pos/all_pos


def do_train(X, Y, areas, area_name='ALM', clf_name='Linear SVC', if_plot=False):
    
    X = X[areas == area_name]
    Y = Y[areas == area_name]

    # Cross validation
    clf = classifiers[clf_name]
    
    pred_accuracy = check_scoring(clf, scoring=None)
    false_negative_rate = check_scoring(clf, scoring=lambda clf, X, Y: false_neg_and_pos(clf, X, Y)[0])
    false_positive_rate = check_scoring(clf, scoring=lambda clf, X, Y: false_neg_and_pos(clf, X, Y)[1])
    scoring={'accuracy': pred_accuracy, 
            'false_pos': false_positive_rate,
            'false_neg': false_negative_rate}
    
    cv_results = cross_validate(clf, X, Y, cv=10,
                                scoring=scoring
                                )
    
    # scores_cv = cross_val_score(clf, X, Y, cv=10)
    accuracy_cv = cv_results['test_accuracy']
    false_negative_cv = cv_results['test_false_neg']
    false_positive_cv = cv_results['test_false_pos']
        
    # All data
    clf.fit(X, Y)
    accuracy_all = clf.score(X, Y)
    false_neg_all, false_pos_all = false_neg_and_pos(clf, X, Y)
    
    # Save results
    score_cv = {'accuracy': accuracy_cv, 'false_neg': false_negative_cv, 'false_pos': false_positive_cv}
    score_all = {'accuracy': accuracy_all, 'false_neg': false_neg_all, 'false_pos': false_pos_all}

    # print(scores_cv)
    print(f'{area_name:>10} @ {clf_name:<20}: ACC {accuracy_cv.mean()*100:4.1f}+/-{accuracy_cv.std()*100:4.1f}%, '
          f'FALSE- {false_negative_cv.mean()*100:4.1f}+/-{false_negative_cv.std()*100:4.1f}%, '
          f'FALSE+ {false_positive_cv.mean()*100:4.1f}+/-{false_positive_cv.std()*100:4.1f}% '
          f'(all ACC {accuracy_all * 100:4.1f}%, F- {false_neg_all * 100:4.1f}%, F+ {false_pos_all * 100:4.1f}%)')
    
    try:
        coef = clf.coefs_ if 'MLP' in clf_name else clf.coef_
    except:
        coef = []

    if if_plot:
        if any([name in clf_name for name in ['MLP', 'Random_forest']]):
            plot_hist_roc(clf.predict_proba(X)[:, 1], Y,
                        score_label='SU probability (SVM)', range=[0, 1])
        else:
            plot_hist_roc(clf.decision_function(X), Y,
                        score_label='Decision function', range=[-20, 5])
            plot_hist_roc(clf.predict_proba(X)[:, 1], Y,
                        score_label='SU probability (SVM)', range=[0, 1])

    return coef, score_cv, score_all, clf


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


def do_train_all(X_scaled, Y, areas):
    
    scores = ['accuracy', 'false_pos', 'false_neg']
    
    cols = pd.MultiIndex.from_product([areas.unique(), classifiers.keys()])
    scores_cv_all = {score: pd.DataFrame(columns=cols) for score in scores}
    clf_all = dict()
    sorters = df_flatten.sorter
    
    for clf_name in classifiers.keys():
        fig = plt.figure(figsize=(15, 10), dpi=300)
        fig.suptitle(clf_name)
        for j, area in enumerate(areas.unique()):
            clf_all[area] = dict()
    
            coef, scores_cv, score_all, clf = do_train(X_scaled, Y, areas,
                                                         area_name=area,
                                                         clf_name=clf_name,
                                                         if_plot=False
                                                         )
            
            for score in scores:
                scores_cv_all[score].loc[:, (area, clf_name)] = scores_cv[score]
            
            clf_all[area][clf_name] = dict(clf=clf, score_all=score_all, coef=coef)
            
            # Plot Venn here
            for i, sorter in enumerate(sorters[areas == area].unique()):
                ax = fig.add_subplot(2, len(areas.unique()), 1 + j + i * len(areas.unique()))
                
                # Do prediction
                this = (areas == area) & (sorters == sorter)
                xx = X_scaled[this]
                yy = df_flatten.label[this]
                yy_predict = clf.predict(xx)
                            
                # Do Venn
                g = venn2([set(np.where(yy)[0]), set(np.where(yy_predict)[0])], set_labels=(sorter[4:], '   fitted'),
                          ax=ax, alpha=0.7)
                ax.set_title(f'{area}, n = {sum(this)}')
            
    return scores_cv_all, clf_all


def plot_cv_results(scores_cv_all):
    
    scores = ['accuracy', 'false_pos', 'false_neg']
    ylims = [(0.7, 1.1), (0, 1), (0, 1)]
    
    for score, ylim in zip(scores, ylims):
        plt.figure(figsize=(15, 10))
        data = scores_cv_all[score].melt(var_name=['Areas','Classfiers'], value_name='10-fold CV')
        ax = sns.boxplot(x='Areas', y='10-fold CV', hue='Classfiers', data=data, showfliers=True)
        # sns.stripplot(x='Areas', y='10-fold CV', hue='Classfiers', data=data,
        #                    dodge=True, jitter=True, color='k', alpha=0.7)
        
        ax.set_title(score)        
        ax.axhline(y=1, ls='--', c='k')
        ax.legend(bbox_to_anchor=(1,1))
        ax.set_ylim(ylim)
    

def plot_venns(clf_all, clf_name='SVC_Linear'):
    #%% All venns for one classifier
    areas = df_flatten.brain_area
    sorters = df_flatten.sorter
    
    for area in areas.unique():
        clf = clf_all[area][clf_name]['clf']
        
        _, axs = plt.subplots(1, len(sorters[areas == area].unique()))
        axs = np.atleast_1d(axs)
        for ax, sorter in zip(axs, sorters[areas == area].unique()):
            # Do prediction
            this = (areas == area) & (sorters == sorter)
            xx = X_scaled[this]
            yy = df_flatten.label[this]
            yy_predict = clf.predict(xx)
                        
            # Do Venn
            g = venn2([set(np.where(yy)[0]), set(np.where(yy_predict)[0])], 
                      ax=ax, set_labels=(sorter, clf_name), alpha=0.8, set_colors=('b', 'g'))
            plt.gcf().suptitle(f'{area}, n = {sum(this)}')
    

# =================================================================================================
# Define classfiers
C = 10
classifiers = {
    'Logistic_L1': LogisticRegression(C=C, penalty='l1',
                                        solver='saga',
                                        multi_class='multinomial',
                                        max_iter=10000,
                                        verbose=False),
    
    'Logistic_L2': LogisticRegression(C=C, penalty='l2',
                                                    solver='saga',
                                                    multi_class='multinomial',
                                                    max_iter=10000,
                                                    verbose=False),
    
    # 'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2',
    #                                         solver='saga',
    #                                         multi_class='ovr',
    #                                         max_iter=10000,
    #                                         verbose=False),
    
    'SVC_Linear': SVC(kernel='linear', C=C, probability=True,
                        random_state=0, verbose=False),
    
    'SVC_RBF': SVC(kernel='rbf', C=C, probability=True,
                        random_state=0, verbose=False),
    
    'MLP_ReLU_20': MLPClassifier(hidden_layer_sizes=[20],
                            activation='relu',
                            #  solver='sgd',
                            early_stopping=False,
                            max_iter=5000,
                            verbose=0),

    'MLP_ReLU_20_20': MLPClassifier(hidden_layer_sizes=[20, 20],
                            activation='relu',
                            #  solver='sgd',
                            early_stopping=False,
                            max_iter=5000,
                            verbose=0),
    
    'Random_forest': RandomForestClassifier(
                                            n_estimators=10,
                                            max_depth=None, 
                                            min_samples_split=2, 
                                            min_samples_leaf=5,
                                            max_features=None,
                                            n_jobs=-1,
                                            random_state=0
                                            )
    # 'GPC': GaussianProcessClassifier(kernel)
}

# classifiers = {
#     # f'Random_forest_{para}': RandomForestClassifier(
#     #                                         n_estimators=10,
#     #                                         max_depth=None, 
#     #                                         min_samples_split=2, 
#     #                                         min_samples_leaf=5,
#     #                                         max_features=None,
#     #                                         n_jobs=-1,
#     #                                         random_state=0)
#     # for para in range(1, 9, 1)
    
#     f'SVC_RBF_{para}': SVC(kernel='rbf', C=para, probability=True,
#                         random_state=0, verbose=False)
#     for para in [0.1, 1, 5, 10, 20, 40, 50, 100, 200, 1000]
# }

# %% Load data
X, Y, X_scaled, scaler, df_flatten = _fetch_manual_label()
areas = df_flatten.brain_area
unique_areas = areas.unique()

# %% One by one
# ['Medulla', 'ALM', 'Thalamus', 'Midbrain', 'Striatum']
area = 'Medulla'
clf_name = 'Random_forest'
coef, scores_cv, score_all, clf = do_train(X_scaled, Y, areas,                                             
                                             area_name=area,
                                             clf_name=clf_name,
                                             if_plot=True,
                                             )

#%% Do all
scores_cv_all, clf_all = do_train_all(X_scaled, Y, areas)
# save to hardisk
# np.savez('export/clf_all.npz', clf_all=clf_all)
# scores_cv_all.to_pickle('export/scores_cv_all.pkl')

# Load previously trained data
# scores_cv_all = pd.read_pickle('export/scores_cv_all.pkl')
# clf_all = np.load('export/clf_all.npz', allow_pickle=True)['clf_all'].tolist()

plot_cv_results(scores_cv_all)
# plot_venns(clf_all, clf_name='Random_forest')


# %%
print(np.vstack([X.columns, coef]).T)

# %% Tsne
aoi = areas != 'ALefefM'
do_tsne(X_scaled[aoi],
        Y[aoi],
        X[aoi], 
        areas[aoi])

