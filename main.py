import os
import mne
import pyvista
import argparse
from utils.load_eeg import load_eeg_data_single_file
from parcellation.load_parcellation_labels import compute_and_save_parcel_signals
from source_reconstruction.beamformer_amoiseev import compute_beamformer_stc
from source_reconstruction.beamformer_lcmv import apply_lcmv_beamformer
from source_reconstruction.forward_solution import read_forward_solution, make_forward_solution, save_forward_solution
from utils.stc_visualization import load_and_morph_stc, plot_brain_flatmap
from config import OUTPUT_DIR, DEFAULT_SUBJECT, SUBJECTS_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_or_compute_stc(raw, fwd, beamformer_method="lcmv", src_type="cortical"):

    stc_fname = os.path.join(OUTPUT_DIR, f"{DEFAULT_SUBJECT}-{src_type}-{beamformer_method}-stc")

    if os.path.exists(f"{stc_fname}-lh.stc"):  # Check if left hemisphere file exists
        print(f"✅ Loading existing source estimate: {stc_fname}")
        return mne.read_source_estimate(stc_fname, subject=DEFAULT_SUBJECT)

    print(f"⚡ No existing source estimate found. Computing {beamformer_method.upper()} beamformer...")

    if beamformer_method == "lcmv":
        stc = apply_lcmv_beamformer(raw, fwd)
    elif beamformer_method == "custom":
        stc = compute_beamformer_stc(raw, fwd, return_stc=True)[0]  # Extract stc from tuple
    else:
        raise ValueError(f"❌ Unknown beamformer method: {beamformer_method}")

    print("💾 Saving computed source estimate...")
    stc.save(stc_fname)
    return stc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Processing Pipeline with Beamformer Selection & Parcellation")
    parser.add_argument("--beamformer", type=str, choices=["lcmv", "custom"], default="lcmv",
                        help="Choose beamformer method: 'lcmv' (default) or 'custom'")
    parser.add_argument("--parc", type=str, default="Schaefer2018_200Parcels_7Networks_order",
                        help="Choose parcellation scheme (default: Schaefer 200).")
    parser.add_argument("--mode", type=str, choices=["pca_flip", "mean", "max"], default="pca_flip",
                        help="Time course extraction mode: 'pca_flip' (default), 'mean', or 'max'.")

    args = parser.parse_args()
    print("🚀 Starting EEG processing pipeline...")
    raw = load_eeg_data_single_file()
    src_type = "cortical"
    fwd = read_forward_solution(src_type)
    if fwd is None:
        print(f"⚡ No existing forward solution found. Computing new forward model for {src_type}...")
        fwd = make_forward_solution(raw, src_type)
        save_forward_solution(fwd, src_type)
    print("✅ Forward solution is ready!")
    stc = read_or_compute_stc(raw, fwd, beamformer_method=args.beamformer, src_type=src_type)
    print("🎉 Beamforming complete! The source estimate is ready for analysis.")
    # ✅ Extract & Save Parcel Signals
    print(f"🔍 Extracting parcel signals using parcellation: {args.parc} with mode: {args.mode}...")
    parcel_signals = compute_and_save_parcel_signals(stc, subject=DEFAULT_SUBJECT, parc=args.parc, mode=args.mode, bem_name=args.beamformer)
    print(f"✅ Parcel signals extracted and saved successfully! Shape: {parcel_signals.shape}")