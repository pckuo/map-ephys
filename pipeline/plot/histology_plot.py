import numpy as np
import datajoint as dj
import itertools
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from pipeline import ephys, ccf, histology


def plot_probe_tracks(session_key, ax=None):
    um_per_px = 20
    # fetch mesh
    vertices, faces = (ccf.AnnotatedBrainSurface
                       & 'annotated_brain_name = "Annotation_new_10_ds222_16bit_isosurf"').fetch1(
        'vertices', 'faces')
    vertices = vertices * um_per_px

    probe_tracks = {}

    if len(session_key) == 1:  # Tracks for one session. Color encodes insertion number
        for probe_insert in (ephys.ProbeInsertion & session_key).fetch('KEY'):
            if not (histology.LabeledProbeTrack & probe_insert):
                continue
            
            if not (histology.LabeledProbeTrack.Point & probe_insert):
                raise ValueError(f'No LabeledProbeTrack.Point for insertion {probe_insert}')
            
            shank_count = (ephys.ProbeInsertion & probe_insert).aggr(
                histology.LabeledProbeTrack.Point, shank_count='COUNT(DISTINCT shank)').fetch1('shank_count')

            all_shank_points = [(histology.LabeledProbeTrack.Point & probe_insert & {'shank': shank_no}).fetch(
                'ccf_x', 'ccf_y', 'ccf_z', order_by='"order"') for shank_no in np.arange(shank_count) + 1]

            probe_tracks[probe_insert['insertion_number']] = [np.vstack(zip(*points)) for points in all_shank_points]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        assert isinstance(ax, Axes3D)

        # cosmetic
        # plt.gca().patch.set_facecolor('white')
        # ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
        # ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
        # ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
        ax.grid(False)
        ax.invert_zaxis()

        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
                        alpha=0.25, lw=0)

        colors = ['r', 'g', 'y', 'b']
        for (k, shank_points), c in zip(probe_tracks.items(), colors):
            for v in shank_points:
                ax.plot(v[:, 0], v[:, 2], v[:, 1], c, label=f'probe {k}')
                
    else:   # Tracks for multiple sessions. Color encodes session
        for session_key_this in session_key.fetch('KEY'):
            probe_tracks[session_key_this['session']] = {}
            
            for probe_insert in (ephys.ProbeInsertion & session_key_this).fetch('KEY', order_by='session'):
                if not (histology.LabeledProbeTrack & probe_insert):
                    continue

                shank_count = (ephys.ProbeInsertion & probe_insert).aggr(
                    histology.LabeledProbeTrack.Point, shank_count='COUNT(DISTINCT shank)').fetch1('shank_count')

                all_shank_points = [(histology.LabeledProbeTrack.Point & probe_insert & {'shank': shank_no}).fetch(
                    'ccf_x', 'ccf_y', 'ccf_z', order_by='"order"') for shank_no in np.arange(shank_count) + 1]

                probe_tracks[session_key_this['session']][probe_insert['insertion_number']] = [np.vstack(zip(*points)) for points in all_shank_points]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        assert isinstance(ax, Axes3D)

        # cosmetic
        # plt.gca().patch.set_facecolor('white')
        # ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
        # ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
        # ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
        ax.grid(False)
        ax.invert_zaxis()

        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
                        alpha=0.25, lw=0)

        # Same color coding as Han's ephys notes
        colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'slateblue', 'blue', 'blueviolet', 'magenta', 'black'] * int(np.ceil(len(probe_tracks)/10))
        for (s, session_tracks), c in zip(probe_tracks.items(), colors):
            for (k, shank_points) in session_tracks.items():
                for v in shank_points:
                    ax.plot(v[:, 0], v[:, 2], v[:, 1], c, label=f'sess {s}' if k == 1 else '')
                
    # ax.legend()
    ax.set_title('Probe Track in CCF (um)')

    return probe_tracks
