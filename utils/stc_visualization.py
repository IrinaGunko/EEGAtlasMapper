import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from nilearn.plotting import plot_glass_brain
from mne.datasets import fetch_fsaverage
from config import SUBJECTS_DIR, OUTPUT_DIR, DEFAULT_SUBJECT, SRC_ICO5
# Fetch fsaverage for visualization
subjects_dir = SUBJECTS_DIR


def load_and_morph_stc(stc, subject=DEFAULT_SUBJECT, title="Beamformer STC"):

    # ✅ Morph STC to fsaverage
    print(f"🔄 Morphing {title} to fsaverage...")
    stc_fs = mne.compute_source_morph(
        stc, subject_from=subject, subject_to="fsaverage",
        subjects_dir=subjects_dir, smooth=5, verbose="error"
    ).apply(stc)
    return stc_fs

def plot_brain_flatmap(stc, annot="HCPMMP1_combined", title="Flatmap Visualization"):
    print(f"🗺️ Visualizing {title} on a flatmap...")
    brain = stc.plot(
        subjects_dir=SUBJECTS_DIR,
        initial_time=0.1,
        clim=dict(kind="value", lims=[3, 6, 9]),
        surface="flat",
        hemi="both",
        size=(1000, 500),
        smoothing_steps=5,
        time_viewer=False,
        add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=10)),
    )
    brain.add_annotation(annot, borders=2)
    return brain

def plot_glass_brain_mri(subject="fsaverage"):
    print("🧠 Visualizing EEG source localization using MRI...")
    misc_path = mne.datasets.misc.data_path()
    fname_T1_electrodes = misc_path / "sample_eeg_mri" / "T1_electrodes.mgz"
    img = nibabel.load(fname_T1_electrodes)  # Load MRI with electrodes
    ras_mni_t = mne.transforms.read_ras_mni_t("fsaverage", SUBJECTS_DIR)
    mni_affine = np.dot(ras_mni_t["trans"], img.affine)
    img_mni = nibabel.Nifti1Image(img.dataobj, mni_affine)  # Convert to MNI space
    plot_glass_brain(
        img_mni,
        cmap="hot_black_bone",
        threshold=0.0,
        black_bg=True,
        resampling_interpolation="nearest",
        colorbar=True,
    )

def compare_stc(stc1, stc2, method1="LCMV", method2="Custom"):
    print("🔍 Comparing STC Peak Activations...")
    # ✅ Find peak activations
    peak_v1, peak_t1 = stc1.get_peak(hemi="lh", vert_as_index=True, time_as_index=True)
    peak_v2, peak_t2 = stc2.get_peak(hemi="lh", vert_as_index=True, time_as_index=True)

    print(f"🔹 {method1} Peak: Vertex {peak_v1}, Time {stc1.times[peak_t1]:.3f}s, Value {stc1.data[peak_v1, peak_t1]:.2f}")
    print(f"🔹 {method2} Peak: Vertex {peak_v2}, Time {stc2.times[peak_t2]:.3f}s, Value {stc2.data[peak_v2, peak_t2]:.2f}")

    # ✅ Overlay peak locations on the brain
    brain1 = plot_brain_flatmap(stc1, title=f"{method1} STC Flatmap")
    brain1.add_foci(stc1.lh_vertno[peak_v1], coords_as_verts=True, hemi="lh", color="blue")

    brain2 = plot_brain_flatmap(stc2, title=f"{method2} STC Flatmap")
    brain2.add_foci(stc2.lh_vertno[peak_v2], coords_as_verts=True, hemi="lh", color="red")

    # ✅ Save Movie for STC Evolution
    print(f"🎥 Saving STC evolution movie for {method1}...")
    brain1.save_movie(time_dilation=20, tmin=0.05, tmax=0.16, interpolation="linear", framerate=10)

    print(f"🎥 Saving STC evolution movie for {method2}...")
    brain2.save_movie(time_dilation=20, tmin=0.05, tmax=0.16, interpolation="linear", framerate=10)

    print("✅ STC Comparison Complete!")
