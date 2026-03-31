"""
Factory Sensor Data Simulator

Generates realistic real-time sensor data for each factory area,
with sinusoidal patterns, random noise, and alert detection.
Designed for WebSocket broadcast to connected dashboard clients.
"""

import asyncio
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Area sensor profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AreaProfile:
    """Defines baseline sensor values and thresholds for a factory area."""
    area_id: str
    name: str
    base_temperature: float      # °C
    base_humidity: float         # %RH
    base_line_speed: float       # units/hour, 0 if not a production area
    base_yield_rate: float       # %, 0 if not applicable
    temp_warning: float          # °C – warning threshold
    temp_critical: float         # °C – critical threshold
    humidity_warning_low: float  # %RH
    humidity_warning_high: float # %RH

AREA_PROFILES: dict[str, AreaProfile] = {
    "lobby": AreaProfile(
        area_id="lobby", name="大廳",
        base_temperature=24.0, base_humidity=50.0,
        base_line_speed=0, base_yield_rate=0,
        temp_warning=30.0, temp_critical=35.0,
        humidity_warning_low=30.0, humidity_warning_high=70.0,
    ),
    "assembly_a": AreaProfile(
        area_id="assembly_a", name="組裝線A",
        base_temperature=25.0, base_humidity=55.0,
        base_line_speed=1250, base_yield_rate=99.9,
        temp_warning=27.5, temp_critical=30.0,
        humidity_warning_low=45.0, humidity_warning_high=65.0,
    ),
    "qc_room": AreaProfile(
        area_id="qc_room", name="品管室",
        base_temperature=23.0, base_humidity=45.0,
        base_line_speed=0, base_yield_rate=99.9,
        temp_warning=28.0, temp_critical=32.0,
        humidity_warning_low=35.0, humidity_warning_high=60.0,
    ),
    "warehouse": AreaProfile(
        area_id="warehouse", name="倉儲區",
        base_temperature=22.0, base_humidity=40.0,
        base_line_speed=0, base_yield_rate=0,
        temp_warning=28.0, temp_critical=33.0,
        humidity_warning_low=30.0, humidity_warning_high=65.0,
    ),
    "conference": AreaProfile(
        area_id="conference", name="會議室",
        base_temperature=24.0, base_humidity=48.0,
        base_line_speed=0, base_yield_rate=0,
        temp_warning=30.0, temp_critical=35.0,
        humidity_warning_low=30.0, humidity_warning_high=70.0,
    ),
}

# ---------------------------------------------------------------------------
# Alert helpers
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    area: str
    type: str       # "temperature" | "humidity"
    level: str      # "warning" | "critical"
    value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "area": self.area,
            "type": self.type,
            "level": self.level,
            "value": round(self.value, 2),
        }


def _check_alerts(area_id: str, profile: AreaProfile, readings: dict[str, float]) -> list[Alert]:
    """Return alerts for any out-of-range readings."""
    alerts: list[Alert] = []

    temp = readings.get("temperature")
    if temp is not None:
        if temp >= profile.temp_critical:
            alerts.append(Alert(area_id, "temperature", "critical", temp))
        elif temp >= profile.temp_warning:
            alerts.append(Alert(area_id, "temperature", "warning", temp))

    hum = readings.get("humidity")
    if hum is not None:
        if hum <= profile.humidity_warning_low or hum >= profile.humidity_warning_high:
            alerts.append(Alert(area_id, "humidity", "warning", hum))

    return alerts


# ---------------------------------------------------------------------------
# SensorSimulator
# ---------------------------------------------------------------------------

class SensorSimulator:
    """
    Generates realistic sensor data for every factory area.

    Values follow a slow sinusoidal drift (period ~10 min) combined with
    Gaussian noise so that consecutive readings look plausible.  Occasional
    random "spikes" are injected at a low probability to exercise alert logic.
    """

    # Sinusoidal drift period in seconds (per sensor type).
    _TEMP_PERIOD = 600.0
    _HUMIDITY_PERIOD = 480.0
    _SPEED_PERIOD = 300.0
    _YIELD_PERIOD = 900.0

    # Amplitude of the sinusoidal component.
    _TEMP_AMP = 1.5       # ±1.5 °C
    _HUMIDITY_AMP = 4.0    # ±4 %RH
    _SPEED_AMP = 60.0      # ±60 units/hour
    _YIELD_AMP = 0.05      # ±0.05 %

    # Gaussian noise σ added on top of sinusoidal pattern.
    _TEMP_NOISE = 0.3
    _HUMIDITY_NOISE = 1.0
    _SPEED_NOISE = 15.0
    _YIELD_NOISE = 0.02

    # Probability per tick that a random spike occurs.
    _SPIKE_PROBABILITY = 0.005

    def __init__(self, profiles: dict[str, AreaProfile] | None = None) -> None:
        self._profiles = profiles or AREA_PROFILES
        # Random phase offsets per area so they don't all oscillate in sync.
        self._phase_offsets: dict[str, float] = {
            aid: random.uniform(0, 2 * math.pi) for aid in self._profiles
        }
        self._start_time = time.monotonic()
        logger.info("SensorSimulator initialised with %d areas", len(self._profiles))

    # -- internal helpers ---------------------------------------------------

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time

    def _sinusoidal(self, period: float, amplitude: float, phase: float) -> float:
        t = self._elapsed()
        return amplitude * math.sin(2 * math.pi * t / period + phase)

    def _maybe_spike(self, base: float, scale: float) -> float:
        """Return a large deviation with low probability, else 0."""
        if random.random() < self._SPIKE_PROBABILITY:
            return random.uniform(2.0, 4.0) * scale * random.choice([-1, 1])
        return 0.0

    # -- public API ---------------------------------------------------------

    def generate_reading(self, area_id: str) -> dict[str, float]:
        """
        Generate a single set of sensor readings for the given area.

        Returns a dict with keys like "temperature", "humidity", and
        optionally "line_speed" / "yield_rate" for production areas.
        """
        profile = self._profiles[area_id]
        phase = self._phase_offsets[area_id]

        # Temperature
        temp = (
            profile.base_temperature
            + self._sinusoidal(self._TEMP_PERIOD, self._TEMP_AMP, phase)
            + random.gauss(0, self._TEMP_NOISE)
            + self._maybe_spike(profile.base_temperature, self._TEMP_AMP)
        )

        # Humidity
        humidity = (
            profile.base_humidity
            + self._sinusoidal(self._HUMIDITY_PERIOD, self._HUMIDITY_AMP, phase + 1.0)
            + random.gauss(0, self._HUMIDITY_NOISE)
        )
        humidity = max(10.0, min(95.0, humidity))  # physical clamp

        readings: dict[str, float] = {
            "temperature": round(temp, 2),
            "humidity": round(humidity, 2),
        }

        # Line speed – only for production areas
        if profile.base_line_speed > 0:
            speed = (
                profile.base_line_speed
                + self._sinusoidal(self._SPEED_PERIOD, self._SPEED_AMP, phase + 2.0)
                + random.gauss(0, self._SPEED_NOISE)
            )
            readings["line_speed"] = round(max(0, speed), 1)

        # Yield rate – for areas that track quality
        if profile.base_yield_rate > 0:
            yld = (
                profile.base_yield_rate
                + self._sinusoidal(self._YIELD_PERIOD, self._YIELD_AMP, phase + 3.0)
                + random.gauss(0, self._YIELD_NOISE)
                + self._maybe_spike(profile.base_yield_rate, 0.15)
            )
            readings["yield_rate"] = round(max(90.0, min(100.0, yld)), 2)

        return readings

    def generate_all(self) -> dict[str, Any]:
        """
        Generate a full sensor payload (all areas) ready for broadcast.

        Returns a dict matching the documented JSON wire format.
        """
        areas: dict[str, dict[str, float]] = {}
        alerts: list[dict[str, Any]] = []

        for area_id, profile in self._profiles.items():
            readings = self.generate_reading(area_id)
            areas[area_id] = readings
            for alert in _check_alerts(area_id, profile, readings):
                alerts.append(alert.to_dict())

        payload: dict[str, Any] = {
            "type": "sensor_update",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "areas": areas,
            "alerts": alerts,
        }
        return payload


# ---------------------------------------------------------------------------
# ConnectionManager  (WebSocket)
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    Manages WebSocket connections for real-time sensor broadcast.

    Enforces a maximum connection limit and provides helpers for
    connecting, disconnecting, and broadcasting JSON payloads.
    """

    MAX_CONNECTIONS = 50

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    @property
    def active_count(self) -> int:
        return len(self._connections)

    async def connect(self, ws: WebSocket) -> bool:
        """
        Accept and register a WebSocket connection.

        Returns True on success, False if the connection limit is reached.
        """
        if len(self._connections) >= self.MAX_CONNECTIONS:
            logger.warning(
                "Connection limit reached (%d). Rejecting new client.",
                self.MAX_CONNECTIONS,
            )
            await ws.close(code=1013, reason="Server busy – max connections reached")
            return False

        await ws.accept()
        self._connections.append(ws)
        logger.info("Client connected. Active connections: %d", len(self._connections))
        return True

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket from the active pool."""
        try:
            self._connections.remove(ws)
        except ValueError:
            pass
        logger.info("Client disconnected. Active connections: %d", len(self._connections))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """
        Send a JSON payload to every connected client.

        Stale / broken connections are silently removed.
        """
        message = json.dumps(payload, ensure_ascii=False)
        stale: list[WebSocket] = []

        for ws in self._connections:
            try:
                await ws.send_text(message)
            except (WebSocketDisconnect, RuntimeError, Exception) as exc:
                logger.debug("Failed to send to client: %s", exc)
                stale.append(ws)

        for ws in stale:
            self.disconnect(ws)


# ---------------------------------------------------------------------------
# Broadcast loop
# ---------------------------------------------------------------------------

async def run_broadcast_loop(
    simulator: SensorSimulator,
    manager: ConnectionManager,
    interval: float = 1.0,
) -> None:
    """
    Continuously generate sensor data and broadcast to all connected
    WebSocket clients at the given interval (default 1 second).

    This coroutine runs indefinitely and should be launched as a
    background task (e.g. via ``asyncio.create_task``).
    """
    logger.info("Broadcast loop started (interval=%.1fs)", interval)
    while True:
        try:
            if manager.active_count > 0:
                payload = simulator.generate_all()
                await manager.broadcast(payload)
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Broadcast loop cancelled – shutting down.")
            break
        except Exception:
            logger.exception("Unexpected error in broadcast loop")
            await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Convenience: standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Print simulated sensor snapshots to stdout (useful for testing)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sim = SensorSimulator()

    async def _demo() -> None:
        for _ in range(5):
            data = sim.generate_all()
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print("---")
            await asyncio.sleep(1)

    asyncio.run(_demo())
