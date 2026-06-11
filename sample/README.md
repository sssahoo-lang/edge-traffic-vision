# Sample media

This folder is a drop-in spot for a local test video (e.g. `traffic.mp4`).

Sample videos aren't checked into the repo (see `.gitignore`). To try the
project without a webcam, place any traffic/street video here and run:

```bash
python main.py --source sample/traffic.mp4
```

A short clip from a public dataset (e.g. a few seconds of dashcam or
street-camera footage) works well for testing the detection and zone-violation
logic.
