import os
import glob
import logging
from concurrent.futures import ProcessPoolExecutor
import mne
from single_processor import EEGProcessor
import config  # Import settings from config.py

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Use paths from config.py
INPUT_DIR = config.EEG_FOLDER
OUTPUT_DIR = config.OUTPUT_DIR

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Processing parameters (can be extended in config.py)
PROCESSING_PARAMS = {
    "bandpass_params": {"l_freq": 0.5, "h_freq": 55, "picks": "eeg", "method": "iir"},
    "notch_params": {"freqs": 50.0, "picks": ["eeg"], "method": "iir"},
    "resample_params": {"sfreq": 256, "npad": "auto", "method": "fft"},
}


def process_file(filepath: str) -> str:

    try:
        filename = os.path.basename(filepath)
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_preprocessed.edf")

        logger.info(f"Processing file: {filepath}")
        raw = mne.io.read_raw_edf(filepath, preload=True)

        # Apply preprocessing pipeline using EEGProcessor
        raw = EEGProcessor.process_raw(
            raw,
            montage="standard_1020",
            bandpass_params=PROCESSING_PARAMS["bandpass_params"],
            notch_params=PROCESSING_PARAMS["notch_params"],
            resample_params=PROCESSING_PARAMS["resample_params"],
        )

        # Save preprocessed file
        raw.export(output_path, fmt="edf", overwrite=True)
        logger.info(f"File saved: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Failed to process file {filepath}: {e}")
        return ""


def main():
    edf_files = glob.glob(os.path.join(INPUT_DIR, "*.edf"))
    if not edf_files:
        logger.warning("No EDF files found in the input directory.")
        return

    logger.info(f"Found {len(edf_files)} EDF files to process.")

    # Process files in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, edf_files))

    # Log completion
    successful_files = [res for res in results if res]
    logger.info(f"Processing complete. {len(successful_files)} files successfully processed.")


if __name__ == "__main__":
    main()
