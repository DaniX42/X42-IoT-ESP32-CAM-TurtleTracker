# ESP32-CAM firmware

The firmware targets the AI Thinker ESP32-CAM and captures a VGA JPEG every five seconds. Each frame is posted to the backend at `/api/frames/{camera_id}`.

## Features

- Wi-Fi station mode
- HTTP JPEG frame upload
- ArduinoOTA updates after the first USB flash
- MQTT telemetry and retained Home Assistant MQTT discovery
- `/health` endpoint on the camera

## Configuration

Copy the values from `firmware/secrets.ini.example` into the local `firmware/secrets.ini`. This file is ignored by Git. Set `api_url` to the reachable backend URL, not `localhost` unless the backend runs on the camera itself.

For the current Mac-hosted backend, use the Mac's LAN address, for example `http://172.20.6.97:8000`. The backend must listen on `0.0.0.0:8000`, and the Mac firewall must allow inbound TCP 8000 from the camera network. The camera also needs outbound TCP access to the MQTT broker on port 1888.

The MQTT broker must be reachable from the camera. Home Assistant discovers these entities automatically:

- Tortoise X and Y in metres
- Tortoise last seen timestamp
- Detection confidence
- Inside-house binary sensor

## First flash

Install PlatformIO, connect the USB-to-serial adapter with GPIO0 held low during reset, then run:

```bash
cd firmware
pio run -t upload --upload-port /dev/cu.usbserial-110
pio device monitor
```

After the device joins Wi-Fi, note its IP address. Subsequent builds can use OTA:

```bash
pio run -t upload --upload-port turtle-cam-outdoor.local
```

Keep the OTA password local and set a strong value in `secrets.ini`.

## Network migration to Proxmox

When the backend moves into its own Proxmox container, change only `api_url` to the container's fixed LAN address, for example `http://192.168.1.50:8000`, then deploy the new firmware once over OTA. The container needs inbound TCP 8000 from the camera VLAN and outbound access to the MQTT broker. No USB access is required after the initial flash.

OTA itself uses UDP 3232 from the development computer to the ESP32. mDNS discovery via `*.local` uses UDP 5353 and may be replaced with the camera's fixed IP if multicast is unavailable.
