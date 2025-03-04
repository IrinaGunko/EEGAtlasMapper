import mne
from pathlib import Path
from config import EEG_FOLDER, EEG_FILE

def load_eeg_data_single_file(eeg_file_path=None, montage_name="standard_1020"):
    if eeg_file_path is None:
        eeg_file_path = Path(EEG_FOLDER) / EEG_FILE

    raw = mne.io.read_raw_edf(eeg_file_path, preload=True)
    if "Status" in raw.ch_names:
        raw.drop_channels(["Status"])
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage)
   # raw.set_eeg_reference(projection=True)
    print(f"✅ EEG data loaded successfully from {Path(eeg_file_path).name} with montage {montage_name}")
    return raw