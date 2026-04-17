"""
mqtt_publisher.py
─────────────────
MQTT publisher — push ผล detection ไป Home Assistant
Topic format: {prefix}/{camera_id}
Payload: JSON
"""

import json
import logging

logger = logging.getLogger(__name__)

# paho-mqtt เป็น optional dependency
try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False
    logger.info("paho-mqtt not installed — MQTT disabled")


class MQTTPublisher:
    """Publish detection results ผ่าน MQTT"""

    def __init__(self, broker: str, port: int = 1883,
                 topic_prefix: str = "occupancy",
                 username: str | None = None,
                 password: str | None = None):

        if not HAS_MQTT:
            raise RuntimeError(
                "paho-mqtt not installed. "
                "Run: pip install 'paho-mqtt>=1.6,<2.0'"
            )

        self.topic_prefix = topic_prefix
        self.client = mqtt.Client()

        if username:
            self.client.username_pw_set(username, password)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self._connected = False

        try:
            self.client.connect(broker, port, keepalive=60)
            self.client.loop_start()
            logger.info(f"MQTT connecting to {broker}:{port}")
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected")
        else:
            logger.error(f"MQTT connect failed: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        logger.warning(f"MQTT disconnected: rc={rc}")

    def publish(self, result) -> bool:
        """Publish 1 detection result"""
        if not self._connected:
            logger.warning("MQTT not connected — skipping publish")
            return False

        topic = f"{self.topic_prefix}/{result.camera_id}"
        payload = json.dumps(result.to_dict())

        info = self.client.publish(topic, payload, qos=1, retain=True)
        if info.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.debug(f"MQTT published: {topic}")
            return True
        else:
            logger.warning(f"MQTT publish failed: {topic} rc={info.rc}")
            return False

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("MQTT stopped")
