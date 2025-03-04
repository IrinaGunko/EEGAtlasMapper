import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
from scipy.stats import pearsonr

class EEGSourceComparator:
    def __init__(self, raw, stc, output_folder="comparison_screenshots"):
        self.raw = raw
        self.stc = stc
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        # Select reference EEG channel (e.g., Fz) and source with max activation
        self.eeg_channel_idx = 10  # Example: Fz
        self.eeg_data = raw.get_data()[self.eeg_channel_idx]
        self.sfreq = raw.info["sfreq"]
        self.ch_names = raw.ch_names

        self.source_idx = np.argmax(stc.data.mean(axis=1))  # Most active source
        self.source_data = stc.data[self.source_idx]
        self.source_sfreq = stc.sfreq

    def compute_psd(self, data, sfreq, title, filename):
        f, psd = welch(data, fs=sfreq, nperseg=256, noverlap=128)
        plt.figure(figsize=(8, 4))
        plt.semilogy(f, psd)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (dB/Hz)")
        plt.title(title)
        plt.grid(True)
        plt.savefig(os.path.join(self.output_folder, filename))
        plt.close()

    def plot_psd_comparison(self):
        self.compute_psd(self.eeg_data, self.sfreq, "EEG PSD (Fz)", "eeg_psd.png")
        self.compute_psd(self.source_data, self.source_sfreq, "Source PSD", "source_psd.png")

    def compute_correlation(self):
        correlation, _ = pearsonr(self.eeg_data[:len(self.source_data)], self.source_data)
        plt.figure(figsize=(6, 4))
        plt.scatter(self.eeg_data[:len(self.source_data)], self.source_data, alpha=0.5)
        plt.xlabel("EEG (Fz) Signal")
        plt.ylabel("Source Estimate Signal")
        plt.title(f"Correlation = {correlation:.3f}")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_folder, "correlation.png"))
        plt.close()
        return correlation

    def plot_topomap_vs_source(self):
        # EEG Topomap
        avg_power = np.abs(self.eeg_data).mean()
        fig, ax = plt.subplots()
        mne.viz.plot_topomap(avg_power * np.ones(len(self.ch_names)), self.raw.info, axes=ax, show=False)
        plt.title("EEG Topomap (Power)")
        plt.savefig(os.path.join(self.output_folder, "eeg_topomap.png"))
        plt.close()
        # Source Activation Map
        brain = self.stc.plot(subject="fsaverage", hemi="both", subjects_dir=None, time_viewer=False)
        brain.screenshot(os.path.join(self.output_folder, "source_activation.png"))
        brain.close()

    def plot_spectrogram(self, data, sfreq, title, filename):
        f, t, Sxx = spectrogram(data, fs=sfreq, nperseg=256, noverlap=128, scaling='density')
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title(title)
        plt.colorbar(label="Power (dB)")
        plt.savefig(os.path.join(self.output_folder, filename))
        plt.close()

    def plot_spectrogram_comparison(self):
        self.plot_spectrogram(self.eeg_data, self.sfreq, "EEG Spectrogram (Fz)", "eeg_spectrogram.png")
        self.plot_spectrogram(self.source_data, self.source_sfreq, "Source Spectrogram", "source_spectrogram.png")

    def run_all_comparisons(self):
        print("🔹 Computing PSD comparison...")
        self.plot_psd_comparison()
        
        print("🔹 Computing EEG-Source Correlation...")
        correlation = self.compute_correlation()
        print(f"✅ Correlation: {correlation:.3f}")

        print("🔹 Computing EEG Topomap vs. Source Map...")
        self.plot_topomap_vs_source()

        print("🔹 Computing Spectrogram comparison...")
        self.plot_spectrogram_comparison()

        print(f"\n✅ All comparisons completed! Screenshots saved in: {self.output_folder}")

