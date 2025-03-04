import mne
import pyvistaqt
from config import SUBJECTS_DIR

def plot_parcellation(parc_name="HCPMMP1", hemisphere="both", highlight_region=None, subjects_dir=None):
    if subjects_dir is None:
        subjects_dir = SUBJECTS_DIR

    print(f"Loading parcellation: {parc_name} for {hemisphere}...")

    hemispheres = ["lh", "rh"] if hemisphere == "both" else [hemisphere]
    labels = {hemi: mne.read_labels_from_annot("fsaverage", parc_name, hemi, subjects_dir=subjects_dir) for hemi in hemispheres}

    print("Initializing brain visualization...")
    Brain = mne.viz.get_brain_class()
    brain = Brain(
        "fsaverage",
        hemi="split" if hemisphere == "both" else hemisphere,
        surf="inflated",
        subjects_dir=subjects_dir,
        cortex="low_contrast",
        background="white",
        size=(1000, 600),
    )

    for hemi in hemispheres:
        brain.add_annotation(parc_name, hemi=hemi)
        print(f"Parcellation {parc_name} added to {hemi} hemisphere.")

    if highlight_region:
        for hemi in hemispheres:
            matched_labels = [label for label in labels[hemi] if label.name.startswith(f"{highlight_region}-{hemi}")]
            if matched_labels:
                brain.add_label(matched_labels[0], hemi=hemi, borders=False)
                print(f"Highlighted region: {highlight_region} on {hemi}")
            else:
                print(f"Warning: Region '{highlight_region}' not found in {parc_name} for {hemi}.")

    print("Displaying visualization...")

    # ✅ Ensure the Qt event loop runs
    plotter = pyvistaqt.BackgroundPlotter()  # This keeps the window open
    plotter.app.exec_()  # Start Qt event loop

    return brain  # Keeps visualization interactive in Jupyter notebooks

# Example usage
if __name__ == "__main__":
    plot_parcellation(parc_name="aparc.a2009s", hemisphere="both")
#aparc.a2009s HCPMMP1 Schaefer2018_400Parcels_17Networks_order