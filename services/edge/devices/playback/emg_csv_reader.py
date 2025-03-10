# File: services/edge/devices/playback/emg_csv_reader.py

import asyncio
import logging
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

from services.common.models.emg import EmgModel

logger = logging.getLogger(__name__)

class EmgCsvReader:
    def __init__(self, csv_path: Path, simulate_live_data=True, loop=True):
        self._csv_path = csv_path
        self._simulate_live_data = simulate_live_data
        self._loop_enabled = loop

        self._data: pd.DataFrame | None = None
        self._num_channels: int | None = None
        self._sampling_rate_hz: float | None = None
        self._sampling_delay_sec: float | None = None
        self._total_duration_sec: float | None = None

    async def load_data(self) -> None:
        """
        Load the CSV data and calculate sampling rate & channels.
        """
        logger.debug(f"📂 Loading CSV from {self._csv_path}...")
        self._data = pd.read_csv(self._csv_path, parse_dates=["timestamp"])
        self._num_channels = len(self._data.columns) - 1  # Subtract 1 for the timestamp column

        # Validate the loaded data
        if self._num_channels <= 0:
            raise ValueError("No EMG channels found.")

        if "timestamp" not in self._data.columns:
            raise ValueError("CSV must contain a 'timestamp' column.")

        if len(self._data) < 2:
            raise ValueError("Not enough rows to compute a sampling rate.")

        self._data["timestamp"] = pd.to_datetime(self._data["timestamp"], format="mixed")
        self._sampling_delay_sec = self._data["timestamp"].iloc[1] - self._data["timestamp"].iloc[0]
        self._sampling_rate_hz = 1.0 / self._sampling_delay_sec.total_seconds()
        self._total_duration_sec = self._data["timestamp"].iloc[-1] - self._data["timestamp"].iloc[0]

        logger.debug(
            f"✅ Loaded {len(self._data)} rows at ~{self._sampling_rate_hz:.2f} Hz, "
            f"{self._num_channels} channels, from '{self._csv_path}'."
        )

    async def emg_reading_loop(self, emg_queue: asyncio.PriorityQueue[EmgModel]) -> None:
        if self._data is None:
            raise RuntimeError("CSV data not loaded—call load_data() first.")

        while True:
            csv_start_timestamp = self._data["timestamp"].iloc[0]
            real_start_timestamp = datetime.now(UTC)

            for _, row in self._data.iterrows():
                emg_data = row[1:].values.astype(float)

                emg_model = EmgModel(
                    timestamp=datetime.now(UTC),
                    data=emg_data.tolist(),
                )

                # Simulate real-time delay
                if self._simulate_live_data:
                    delay_from_csv_start = row["timestamp"] - csv_start_timestamp
                    delay_from_real_start = datetime.now(UTC) - real_start_timestamp
                    sleep_duration = max(0,
                                         delay_from_csv_start.total_seconds() - delay_from_real_start.total_seconds())
                    await asyncio.sleep(sleep_duration)

                emg_queue.put_nowait(emg_model)

            if not self._loop_enabled:
                logger.info("🔁 Playback finished once; not looping.")
                break
            else:
                logger.info("🔁 Looping CSV playback again...")