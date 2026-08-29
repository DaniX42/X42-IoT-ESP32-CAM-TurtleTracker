import json
import logging

logger = logging.getLogger(__name__)


class MqttPublisher:
    """Optional MQTT publisher for the door house-state; a no-op when disabled or unreachable."""

    def __init__(self, host: str | None, port: int, user: str | None, password: str | None, topic_prefix: str, enabled: bool):
        self.topic_prefix = topic_prefix.rstrip("/")
        self._client = None
        if not enabled or not host:
            return
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.warning("paho-mqtt is not installed; MQTT publishing disabled")
            return
        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        if user:
            client.username_pw_set(user, password)
        try:
            client.connect(host, port)
            client.loop_start()
        except OSError:
            logger.warning("Could not connect to MQTT broker %s:%s", host, port)
            return
        self._client = client

    def publish_state(self, inside_house: bool) -> None:
        if self._client is None:
            return
        self._client.publish(f"{self.topic_prefix}/state", "inside" if inside_house else "outside", retain=True)

    def publish_event(self, event: str) -> None:
        if self._client is None:
            return
        self._client.publish(f"{self.topic_prefix}/event", event)

    def publish_position(self, x: float, y: float, speed: float, confidence: float) -> None:
        """Publish turtle position coordinates and motion metrics."""
        if self._client is None:
            return
        payload = json.dumps({"x": x, "y": y, "speed": speed, "confidence": confidence})
        self._client.publish(f"{self.topic_prefix}/position", payload, retain=True)
