"""Redis Cache and Real-Time Buffer for Live Telemetry and Sequences."""

import json
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
import redis

from src.config import settings
from src.utils.logger import logger


class TelemetryCache:
    """Manages real-time sliding sequence buffers for turbofan engines."""

    def __init__(self, redis_url: str = settings.REDIS_URL, max_buffer_len: int = 40):
        self.redis_url = redis_url
        self.max_buffer_len = max_buffer_len
        self.client: Optional[redis.Redis] = None
        self.use_memory_fallback = False
        self._memory_store: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_buffer_len))

        self._connect_redis()

    def _connect_redis(self):
        """Attempts to connect to Redis server; falls back to in-memory deque if unavailable."""
        try:
            r = redis.from_url(self.redis_url, decode_responses=True, socket_timeout=1.0)
            r.ping()
            self.client = r
            logger.info(f"Connected to Redis cache at {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}). Using in-memory cache fallback for local development.")
            self.use_memory_fallback = True
            self.client = None

    def push_reading(self, engine_id: int, reading: Dict[str, Any]):
        """Pushes a new sensor telemetry point into the engine's sliding window."""
        if self.use_memory_fallback or self.client is None:
            self._memory_store[engine_id].append(reading)
        else:
            try:
                key = f"engine:{engine_id}:buffer"
                self.client.rpush(key, json.dumps(reading))
                self.client.ltrim(key, -self.max_buffer_len, -1)
            except Exception as e:
                logger.debug(f"Redis write error ({e}), storing in memory.")
                self._memory_store[engine_id].append(reading)

    def get_recent_readings(self, engine_id: int, count: int = 30) -> List[Dict[str, Any]]:
        """Retrieves the last `count` readings for an engine."""
        if self.use_memory_fallback or self.client is None:
            items = list(self._memory_store[engine_id])
            return items[-count:] if len(items) >= count else items
        else:
            try:
                key = f"engine:{engine_id}:buffer"
                raw_items = self.client.lrange(key, -count, -1)
                return [json.loads(item) for item in raw_items]
            except Exception as e:
                logger.debug(f"Redis read error ({e}), reading from memory fallback.")
                items = list(self._memory_store[engine_id])
                return items[-count:] if len(items) >= count else items

    def is_connected(self) -> bool:
        """Returns True if connected to an external Redis instance."""
        if self.client is not None:
            try:
                return self.client.ping()
            except Exception:
                return False
        return False


cache = TelemetryCache()
