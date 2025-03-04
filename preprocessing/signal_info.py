import mne
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalInfo:
    @staticmethod
    def get(raw: mne.io.Raw, log_prefix: str = None) -> dict:

        info = {
            "sampling_frequency": raw.info["sfreq"],
            "highpass_cutoff": raw.info["highpass"],
            "lowpass_cutoff": raw.info["lowpass"],
            "channel_count": raw.info["nchan"],
            "bad_channels": raw.info["bads"],
            "meas_date": raw.info["meas_date"],
            "description": raw.info.get("description", "No description available")
        }
        if log_prefix:
            logger.info(f"{log_prefix}: {info}")
        return info
