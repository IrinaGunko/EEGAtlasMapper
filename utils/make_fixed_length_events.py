import mne
import os
from utils.load_eeg import load_eeg_data_single_file
from config import PROJ_DIR

# Define paths
OUTPUT_DIR = os.path.join(PROJ_DIR, "output_epochs")  # Directory to save epoch FIF files
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output folder if it doesn't exist

# Load the preprocessed EEG
raw = load_eeg_data_single_file()

# Define epoch length (20 seconds)
epoch_length = 20  # seconds
sfreq = raw.info["sfreq"]  # Sampling frequency
n_samples = int(epoch_length * sfreq)  # Convert to samples

# Create events at fixed intervals (every 20 seconds)
event_id = {"epoch_start": 1}
events = mne.make_fixed_length_events(raw, id=1, start=0, stop=None, duration=epoch_length)

# Create epochs
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=epoch_length, baseline=None, preload=True)

print(f"Created {len(epochs)} epochs, each {epoch_length} seconds long.")

# Save each epoch as a separate FIF file
for i in range(len(epochs)):
    epoch_filename = os.path.join(OUTPUT_DIR, f"epoch_{i+1:03d}.fif")  # epoch_001.fif, epoch_002.fif, etc.
    epochs[i:i+1].save(epoch_filename, overwrite=True)  # ✅ Corrected slicing to keep MNE structure
    print(f"Saved {epoch_filename}")

print("✅ All epochs saved successfully!")
