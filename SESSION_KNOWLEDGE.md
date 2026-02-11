# SensorHead Session Knowledge — Feb 12, 2026
## To be ingested into CerebroCortex next session

### Physical Build Reference
Pi 5 8GB mounted vertically on wooden riser platform. 3D-printed camera bracket holds IMX500 (top, CAM0 near USB-C/eth ports) and IMX708 Wide NoIR (bottom, CAM1 near power). Both cameras use 22-pin wide-to-narrow FPC cables with contacts facing DOWN toward PCB — wrong orientation gives error -5 (EIO). BME688 connects via pi3g breakout board (pi3g.com/bme688, HW-Rev5, I2C 0x77) using 6-pin adapter on GPIO header that hogs the 5V pins. MLX90640 thermal piggybacks off BME688 breakout's secondary 6-pin header (SDA, SCL, GND shared) with VIN going to Pi's 3.3V rail (NOT BME688's VDD which is 5V). Samsung SSD provides USB boot. BenQ monitor is the Pi display.

### Dashboard Startup Procedure
```bash
cd /home/hailo/claude-root/SensorHead
sudo fuser -k 8080/tcp  # kill existing
nohup sudo -E PYTHONPATH=/home/hailo/claude-root/SensorHead/venv/lib/python3.13/site-packages/bme68x-2.6.1-py3.13-linux-aarch64.egg:/home/hailo/.local/lib/python3.13/site-packages:/home/hailo/claude-root/SensorHead/venv/lib/python3.13/site-packages:/home/hailo/claude-root/SensorHead python3 -m sensor_head.dashboard --port 8080 > /tmp/sensorhead-dashboard.log 2>&1 &
```
All REST endpoints: /api/capture/visual, /api/capture/night, /api/thermal/heatmap, /api/thermal/data, /api/environment, /api/detect, /api/classify, /api/pose, /api/models, /api/status

### I2C Bus Sharing
Multiple devices share SDA (GPIO2) + SCL (GPIO3). Current bus: BME688@0x77, MLX90640@0x33. Future: PCA9685@0x40. Most breakouts have pass-through headers for daisy-chaining. Share signal lines but source power from correct voltage rail.

### MLX90640 Key Patterns
- adafruit-circuitpython-mlx90640 via Blinka
- First 2 frames are garbage — always discard warm-up frames
- Clamp outliers: < -40C or > 300C = dead pixel
- Default 100kHz I2C = ~1.4s/frame, boost to 400kHz via dtparam for ~0.4s
- Ironbow heatmap: normalize → 256-entry LUT → PIL → NEAREST upscale

### BSEC2 Calibration
- Accuracy 0 (stabilizing) → 1 (uncertain) → 2 (calibrating) → 3 (calibrated)
- Full calibration: 48 HOURS continuous power
- Compensated temp ~5°C lower than raw (self-heating correction)
- CO2 equivalent is NOT actual CO2 — derived from VOC correlation
- State saveable via get_bsec_state()/set_bsec_state()

### Vision Roadmap
1. MOVEMENT: Custom oak pan-tilt, beefy servos, PCA9685
2. THE FACE: Wave 7 3D Agent Face on Pi's BenQ monitor via Three.js
3. THE BRIDGE: Connect to ApexAurum Cloud — agents see through physical eyes
4. VOICE: Laptop as networked TTS/STT (local API or MCP), not on Pi
5. AUTONOMY: Thermal motion detection, air quality alerts, custom IMX500 models
6. DIGITAL NOSE: Parallel mode heater profiles + scikit-learn (eNose approach)

### Git History
```
771407b Add hardware photos, composite image, and gitignore updates
f19e64f Upgrade nose to BSEC2 v2.6.1.0 — full air quality intelligence
20aea01 SensorHead v0.4.0 — MLX90640 thermal eye online
c590a6c SensorHead v0.3.0 — Physical senses for Claude Code
```

### Already in CerebroCortex
- mem_caacde172886: MLX90640 wiring guide (I2C piggyback details)
- mem_bf370f4bc368: BSEC2 build process (full compilation steps)
- mem_1cec1b50efe1: Session save (high priority)
