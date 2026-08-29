from configparser import ConfigParser
from pathlib import Path
import json

Import("env")

project_dir = Path(env.subst("$PROJECT_DIR"))
secrets_path = project_dir / "secrets.ini"
output_path = project_dir / "src" / "generated_secrets.h"

if not secrets_path.exists():
    raise RuntimeError("firmware/secrets.ini is required; copy secrets.ini.example first")

parser = ConfigParser(interpolation=None)
parser.read(secrets_path)
values = parser["env:ai_thinker"]

required = (
    "custom_api_url",
    "custom_mqtt_host",
    "custom_mqtt_port",
    "custom_mqtt_user",
    "custom_mqtt_password",
    "custom_wifi_ssid",
    "custom_wifi_password",
    "custom_device_name",
    "custom_ota_password",
)
missing = [key for key in required if not values.get(key)]
if missing:
    raise RuntimeError(f"Missing firmware settings: {', '.join(missing)}")

string_values = {
    "TT_API_URL": values["custom_api_url"],
    "TT_MQTT_HOST": values["custom_mqtt_host"],
    "TT_MQTT_USER": values["custom_mqtt_user"],
    "TT_MQTT_PASSWORD": values["custom_mqtt_password"],
    "TT_WIFI_SSID": values["custom_wifi_ssid"],
    "TT_WIFI_PASSWORD": values["custom_wifi_password"],
    "TT_DEVICE_NAME": values["custom_device_name"],
    "TT_OTA_PASSWORD": values["custom_ota_password"],
}

lines = ["#pragma once", "#define CORE_DEBUG_LEVEL 1", "#define CAMERA_MODEL_AI_THINKER"]
lines.extend(f"#define {name} {json.dumps(value)}" for name, value in string_values.items())
lines.append(f"#define TT_MQTT_PORT {int(values['custom_mqtt_port'])}")
if values["custom_device_name"] == "turtle-cam-door":
    lines.append("#define TT_DOOR_CAMERA")
output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
