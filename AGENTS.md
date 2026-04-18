# Agent Instructions

- Start with [README.md](README.md), [app.py](app.py), [datacollection.py](datacollection.py), and [requirements.txt](requirements.txt) before making changes.
- This project is a webcam-driven Flask app. `app.py` loads `models/keras_model.h5` and `models/labels.txt` at import time, opens `cv2.VideoCapture(0)`, and builds the meme map from `static/img/monekey/*`.
- Deployment is container-friendly through the root `Dockerfile`; `HOST`, `PORT`, `FLASK_DEBUG`, and `CAMERA_INDEX` control runtime behavior.
- Keep the JSON contract used by the front end stable: `/stats` returns `meme`, `meme_path`, and `hands`, and `/toggle_camera` returns `enabled`.
- Keep the template and API in sync. [templates/index.html](templates/index.html) polls `/stats` on a short interval and expects the camera feed and meme image paths to remain valid.
- Treat [datacollection.py](datacollection.py) as a separate utility for collecting hand crops. Its `SAVE_DIR` is hardcoded and Windows-specific, so do not assume it is portable without changing it.
- Preserve the current asset layout under `static/img/monekey/` unless you update every hardcoded reference. The typo in that folder name is part of the current contract.
- Prefer small, targeted edits over broad refactors. Avoid reformatting unrelated files.
- There is no automated test suite in the repository. Verify runtime changes manually by launching the app with the virtual environment active and checking the browser, webcam, and model loading behavior.
- Update [README.md](README.md) only when setup, runtime usage, or project structure changes.