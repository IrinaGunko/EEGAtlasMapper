import os
import mne
import pyvistaqt
from config import SUBJECTS_DIR, TRANS, SRC_ICO5, BEM
from utils.load_eeg import load_eeg_data_single_file

def visualize_coregistration(raw, subject="fsaverage", subjects_dir=SUBJECTS_DIR, trans=TRANS):
    """Check coregistration: head position relative to EEG sensors and MRI."""
    print(f"📌 Checking coregistration for subject: {subject}")

    fig = mne.viz.plot_alignment(
        raw.info,
        subject=subject,
        subjects_dir=subjects_dir,
        trans=trans,
        src = SRC_ICO5,
        surfaces=("white", "outer_skin", "inner_skull", "outer_skull"),
        bem=BEM,
        coord_frame="mri",
        mri_fiducials=True,
        show_axes=True,
    )

    mne.viz.set_3d_view(fig, azimuth=25, elevation=70, focalpoint=[0, -0.005, 0.01])

    print("✅ Coregistration visualization ready.")
    return fig


def visualize_source_space(subject="fsaverage", subjects_dir=SUBJECTS_DIR, src_type="cortical"):
    """Plot the cortical or volume-based source space."""
    print(f"📌 Visualizing {src_type} source space for subject: {subject}")

    if src_type == "cortical":
        src = mne.read_source_spaces(SRC_ICO5)
    else:
        print("❌ Invalid src_type. Use 'cortical' or 'volumetric'.")
        return None

    print("✅ Source space loaded. Launching 3D visualization...")

    # ✅ 3D Visualization for Source Space
    fig = mne.viz.plot_alignment(
        subject=subject,
        subjects_dir=subjects_dir,
        surfaces="white",
        coord_frame="mri",
        src=src,
    )

    mne.viz.set_3d_view(
        fig,
        azimuth=173.78,
        elevation=101.75,
        distance=0.30,
        focalpoint=(-0.03, -0.01, 0.03),
    )

    print(f"✅ {src_type} source space visualization ready.")

    return fig


if __name__ == "__main__":
    print("🚀 Starting forward model visualization...")

    # ✅ Load EEG data
    raw = load_eeg_data_single_file()
    raw.set_montage("standard_1020") 

    # ✅ Initialize PyVista before visualization
    mne.viz.set_3d_backend("pyvistaqt")

    # ✅ Check coregistration (EEG <-> MRI)
    visualize_coregistration(raw)

    # ✅ Visualize source space
    visualize_source_space()

    print("🎉 Visualization complete. Adjust views as needed!")

    # ✅ Keep PyVista window open
    pyvistaqt.BackgroundPlotter().app.exec_()
