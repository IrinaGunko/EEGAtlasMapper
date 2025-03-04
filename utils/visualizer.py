import mne
import matplotlib.pyplot as plt
import os
import datetime
from config import OUTPUT_DIR

class EEGVisualizer:
    def __init__(self, raw):
        self.raw = raw

    def plot_eeg_alignment(self, trans, src, save_name=None):
        fig = mne.viz.plot_alignment(
            self.raw.info,
            src=src,
            eeg=["original", "projected"],
            trans=trans,
            show_axes=True,
            mri_fiducials=True,
            dig="fiducials",
        )
        self._save_figure(fig, save_name)
        self._wait_for_close()

    def plot_mri_alignment(self, trans, src, subject, subjects_dir, bem=None, save_name=None):
        surfaces = ("white", "outer_skin", "inner_skull", "outer_skull") if bem else ("white", "outer_skin")
        fig = mne.viz.plot_alignment(
            self.raw.info,
            subject=subject,
            subjects_dir=subjects_dir,
            trans=trans,
            src=src,
            bem=bem,
            coord_frame="mri",
            mri_fiducials=True,
            show_axes=True,
            surfaces=surfaces,
        )
        mne.viz.set_3d_view(fig, 25, 70, focalpoint=[0, -0.005, 0.01])
        self._save_figure(fig, save_name)
        self._wait_for_close()

import mne
import matplotlib.pyplot as plt
import os
import datetime
from config import OUTPUT_DIR

class EEGVisualizer:
    def __init__(self, raw):
        self.raw = raw

    def plot_eeg_alignment(self, trans, src, save_name=None):
        fig = mne.viz.plot_alignment(
            self.raw.info,
            src=src,
            eeg=["original", "projected"],
            trans=trans,
            show_axes=True,
            mri_fiducials=True,
            dig="fiducials",
        )
        self._save_figure(fig, save_name)
        self._wait_for_close()

    def plot_mri_alignment(self, trans, src, subject, subjects_dir, bem=None, save_name=None):
        surfaces = ("white", "outer_skin", "inner_skull", "outer_skull") if bem else ("white", "outer_skin")
        fig = mne.viz.plot_alignment(
            self.raw.info,
            subject=subject,
            subjects_dir=subjects_dir,
            trans=trans,
            src=src,
            bem=bem,
            coord_frame="mri",
            mri_fiducials=True,
            show_axes=True,
            surfaces=surfaces,
        )
        mne.viz.set_3d_view(fig, 25, 70, focalpoint=[0, -0.005, 0.01])
        self._save_figure(fig, save_name)
        self._wait_for_close()

    def plot_source_space_2d(self, src, save_name=None, **plot_bem_kwargs):
        fig_bem = mne.viz.plot_bem(src=src, **plot_bem_kwargs)
        self._save_figure(fig_bem, f"{save_name}_2D" if save_name else None)
        self._wait_for_close()

    def plot_source_space_3d(self, src, subject, subjects_dir, save_name=None):
        fig_3d = mne.viz.plot_alignment(
            subject=subject,
            subjects_dir=subjects_dir,
            surfaces="white",
            coord_frame="mri",
            src=src,
        )
        mne.viz.set_3d_view(
            fig_3d,
            azimuth=173.78,
            elevation=101.75,
            distance=0.30,
            focalpoint=(-0.03, -0.01, 0.03),
        )
        self._save_figure(fig_3d, f"{save_name}_3D" if save_name else None)
        self._wait_for_close()

    def _save_figure(self, fig, save_name):
        """Saves the figure as an image with a timestamp."""
        if save_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_name}_{timestamp}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            fig.savefig(save_path, dpi=300)
            print(f"✅ Visualization saved: {save_path}")

    def _wait_for_close(self):
        plt.show(block=True)


    def _save_figure(self, fig, save_name):
        if save_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_name}_{timestamp}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            fig.savefig(save_path, dpi=300)
            print(f"✅ Visualization saved: {save_path}")

    def _wait_for_close(self):
        plt.show(block=True)
