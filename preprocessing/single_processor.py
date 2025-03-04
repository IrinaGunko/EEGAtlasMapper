import logging
from typing import Optional
import mne
from bandpass_filter import BandpassFilter
from notch_filter import NotchFilter
from signal_info import SignalInfo
from resample import ResampleData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EEGProcessor:
    @staticmethod
    def process_raw(
        raw: mne.io.Raw,
        montage: str = "standard_1020",
        bandpass_params: Optional[dict] = None,
        notch_params: Optional[dict] = None,
        resample_params: Optional[dict] = None,
        extralogging: bool = False
    ) -> mne.io.Raw:

        try:
            # Step 1: Set Montage
            logger.info("Setting montage: %s", montage)
            raw.set_montage(montage)
            if extralogging: SignalInfo.get(raw, "Signal Info after montage")

            # Step 2: Apply Bandpass Filter
            if bandpass_params:
                logger.info("Applying bandpass filter with parameters: %s", bandpass_params)
                raw = BandpassFilter.apply(raw, **bandpass_params)
                if extralogging: SignalInfo.get(raw, "Signal Info after bandpass")

            # Step 3: Apply Notch Filter
            if notch_params:
                logger.info("Applying notch filter with parameters: %s", notch_params)
                raw = NotchFilter.apply(raw, **notch_params)
                if extralogging: SignalInfo.get(raw, "Signal Info after notch")

            # Step 4: Resample
            if resample_params:
                logger.info("Applying resampling with parameters: %s", resample_params)
                raw = ResampleData.apply(raw, **resample_params)
                if extralogging: SignalInfo.get(raw, "Signal Info after resample")

        except Exception as e:
            logger.error("Error during processing: %s", e)
            raise

        logger.info("EEG processing completed successfully.")
        return raw