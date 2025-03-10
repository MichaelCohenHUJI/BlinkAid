# File: services/edge/devices/playback/playback_device.py

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterable

from services.common.models.emg import EmgModel
from services.common.models.stimulation import StimulationModel
from services.edge.devices.base_device import BaseDevice
from services.edge.devices.device_info import DeviceInfo
from services.edge.devices.device_type import DeviceType
from services.edge.devices.playback.emg_csv_reader import EmgCsvReader
from services.edge.devices.playback.recordings import PLAYBACK_RECORDINGS_DIR
from services.edge.emg_processing.emg_filters.emg_scipy_filters import LowPassFilter

logger = logging.getLogger(__name__)

class PlaybackDevice(BaseDevice):
    """
    Plays back EMG samples from a CSV to simulate a real device.
    """

    @staticmethod
    async def discover(
            recordings_dir: Path = PLAYBACK_RECORDINGS_DIR,
            timeout_sec=3) -> AsyncIterable[DeviceInfo]:
        logger.debug(f"🔎 Searching for CSV recordings in '{recordings_dir}'...")
        recordings = list(recordings_dir.glob("*.csv"))
        for recording in recordings:
            yield DeviceInfo(
                type=DeviceType.PLAYBACK,
                address=str(recording),
                name=recording.stem)

    def __init__(
            self,
            device_info: DeviceInfo,
            emg_queue: asyncio.PriorityQueue[EmgModel],
            command_queue: asyncio.Queue[StimulationModel],
            simulate_live_data=True, loop=True
    ):
        super().__init__(device_info, emg_queue, command_queue)

        self._csv_reader = EmgCsvReader(Path(device_info.address), simulate_live_data, loop)
        self._is_running = False

    @property
    def num_channels(self) -> int:
        if self._csv_reader._num_channels is None:
            raise RuntimeError("Device not connected yet—no channel info.")
        return self._csv_reader._num_channels

    @property
    def sampling_rate_hz(self) -> float:
        if self._csv_reader._sampling_rate_hz is None:
            raise RuntimeError("Device not connected yet—no sampling rate.")
        return self._csv_reader._sampling_rate_hz

    @property
    def default_filters(self):
        if self._csv_reader._num_channels is None or self._csv_reader._sampling_rate_hz is None:
            raise RuntimeError("Device not connected yet—no channel info or sampling rate.")
        return [
            LowPassFilter(
                num_channels=self.num_channels,
                fs=self.sampling_rate_hz,
                cutoff=10)
        ]

    async def _emg_reading_loop(self) -> None:
        await self._csv_reader.emg_reading_loop(self._emg_queue)

    async def _commands_handling_loop(self) -> None:
        while self._is_running:
            command = await self._commands_queue.get()
            logger.info(f"⚡️ Playback device ignoring command: {command}")
            self._commands_queue.task_done()

    async def __aenter__(self):
        await self._csv_reader.load_data()
        self._is_running = True
        self._tasks = [
            asyncio.create_task(self._emg_reading_loop()),
            asyncio.create_task(self._commands_handling_loop()),
        ]

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._is_running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return False