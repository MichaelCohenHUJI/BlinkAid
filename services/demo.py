import asyncio
import logging
import time
from pathlib import Path

from services.detection.emg_detectors.base_emg_detector import BaseEmgDetector
from services.detection.emg_detectors.examples.blink_detector_cnn import BlinkDetectorCNN
from services.detection.emg_detectors.examples.blink_detector_threshold_voting import BlinkDetectorThresholdVoting
from services.edge.devices.playback.emg_csv_reader import EmgCsvReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


async def detection_demo(detector: BaseEmgDetector, csv_path: Path):
    emg_queue = asyncio.Queue()
    detections = []

    logger.info(f"📂 Loading CSV from {csv_path}...")
    emg_reader = EmgCsvReader(csv_path, simulate_live_data=False, loop=False)
    await emg_reader.load_data()

    # Start reading EMG data and running the detector
    async def detector_loop():
        start_time = time.time()
        logger.info("🔍 Starting detection loop...")

        while True:
            try:
                emg_model = await asyncio.wait_for(emg_queue.get(), timeout=1)
                detection = detector.detect(emg_model)
                if detection:
                    logger.info(f"Detection: {detection}")
                    detections.append(detection)
                emg_queue.task_done()

            except asyncio.TimeoutError:
                logger.info("🛑 Detection loop timed out.")
                break

        logger.info(f"🛑 Detection loop finished in {time.time() - start_time:.2f} seconds.")
        logger.info(f"🔍 Detected {len(detections)} events.")

    # Run the EMG reading loop and processing loop concurrently
    await asyncio.gather(
        emg_reader.emg_reading_loop(emg_queue),
        detector_loop()
    )


if __name__ == "__main__":
    # <-- USE YOUR DETECTOR HERE -->
    # detector = BlinkDetectorThresholdVoting()
    detector = BlinkDetectorCNN()

    # <-- USE YOUR CSV HERE -->
    data_path = PROJECT_ROOT / "data/michael_3-3/2025_03_03_1350_michael_blinks.csv"

    asyncio.run(detection_demo(detector, data_path))
