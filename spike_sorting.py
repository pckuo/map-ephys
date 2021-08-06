# %%
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

import datajoint as dj
from pipeline import ephys, lab, experiment, ccf

# %%
# == Get queries ==
all_unit = ((ephys.Unit.proj('unit_amp', 'unit_snr')
             & (ephys.ProbeInsertion & ephys.UnitNote).proj())  # All units from all sorted sessions
            * ephys.UnitStat * ephys.ClusterMetric).proj(..., _='unit_quality')
# * ephys.MAPClusterMetric.DriftMetric)   # DriftMetric is incomplete

sorters = (dj.U('note_source') & ephys.UnitNote).fetch('note_source')
session_sorter = (dj.U('subject_id', 'session',
                  'insertion_number', 'note_source') & ephys.UnitNote)

all_sorters = all_unit.proj()

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

    all_sorters = all_sorters * this_sorter

# %% == Get metrics ==
keys = ['subject_id', 'session', 'insertion_number', 'unit']
xs = ['unit_amp', 'unit_snr', 'isi_violation', 'avg_firing_rate', 'presence_ratio', 'amplitude_cutoff',
      'isolation_distance', 'l_ratio', 'd_prime', 'nn_hit_rate', 'nn_miss_rate',
      'max_drift',
      'cumulative_drift',
      #   'drift_metric',
      ]

print('fetching...', end='')
df_all = (all_unit * all_sorters).proj(*xs, *sorters).fetch(format='frame').astype('float')
print('done!')

# Deal with missing values (only keep units with all non-nan metrics)
np.sum(np.all(~np.isnan(df_all), axis=1))
isnan = np.sum(np.isnan(df_all), axis=0)
xs_allnotnan = isnan[isnan == 0].index

# Training data set
df = df_all[list(xs_allnotnan) + list(sorters)]

# Summary of all labels
print(df.groupby(['sortDave', 'sortHan', 'sortSusu'], dropna=False).size().reset_index(name='Count'))


# %%
# == Get labels ==
