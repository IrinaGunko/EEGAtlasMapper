import mne
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from config import SUBJECTS_DIR, MNI152

# Define file paths
MNI152_FILE = f"{MNI152}/MNI152_T1_1mm_brain.nii.gz"

def plot_fsaverage_head():
    """Load and plot the FreeSurfer fsaverage head model."""
    print("🔹 Loading FreeSurfer fsaverage head model...")
    
    fig = mne.viz.plot_bem(
        subject="fsaverage",
        subjects_dir=SUBJECTS_DIR,
        orientation="coronal",
        slices=[50, 100, 150],  # Select slices to visualize
        brain_surfaces="white"
    )
    
    return fig

def plot_mni152_head():
    """Load and plot the MNI152 head model."""
    print(f"🔹 Loading MNI152 template from {MNI152_FILE}...")

    # Load MNI152 MRI file
    mni_mri = nib.load(MNI152_FILE)
    mni_data = mni_mri.get_fdata()

    # Plot a coronal slice
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.rot90(mni_data[:, :, mni_data.shape[2] // 2]), cmap="gray")
    ax.set_title("MNI152 Head Model (Coronal Slice)")
    ax.axis("off")

    return fig

def compare_fsaverage_vs_mni152():
    """Compare the fsaverage and MNI152 head models side by side."""
    print("🚀 Comparing FreeSurfer fsaverage vs. MNI152 Head Model...")
    
    fig_fs = plot_fsaverage_head()
    fig_mni = plot_mni152_head()

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Convert FreeSurfer figure to image for side-by-side comparison
    fig_fs.canvas.draw()
    axes[0].imshow(fig_fs.canvas.buffer_rgba())
    axes[0].set_title("FreeSurfer fsaverage Head Model")
    axes[0].axis("off")

    # Convert MNI152 figure to image
    fig_mni.canvas.draw()
    axes[1].imshow(fig_mni.canvas.buffer_rgba())
    axes[1].set_title("MNI152 Head Model")
    axes[1].axis("off")

    plt.show()

if __name__ == "__main__":
    compare_fsaverage_vs_mni152()
