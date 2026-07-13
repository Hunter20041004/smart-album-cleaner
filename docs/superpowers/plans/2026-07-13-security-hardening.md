# Smart Album Cleaner Security Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the local photo workflow working while confining filesystem access, removing vulnerable dependencies, and making model loading safe.

**Architecture:** A canonical-root registry protects file routes. The service binds to loopback. A focused MediaPipe Tasks adapter replaces legacy solutions, and all PyTorch loads use weights-only mode. Model assets are checksum verified.

**Tech Stack:** Python 3.11, FastAPI, PyTorch, MediaPipe Tasks, Vue/Vite, pytest.

## Global Constraints

- Modify only the Codex GitHub clone/worktree; never touch the user's original project or photographs.
- Tests use temporary synthetic images and temporary directories only.
- Preserve frontend API response shapes and scan/trash/restore behavior.
- Follow one Red → Green → Refactor slice at a time.

---

### Task 1: Confine filesystem access

**Files:**
- Create: `tests/test_backend_security.py`
- Modify: `backend/main.py`
- Modify: `requirements-dev.txt`

**Interfaces:**
- Produces: `authorize_root(path)`, `require_authorized_path(path, root=None) -> Path`, and an in-memory canonical root set.

- [ ] **Step 1: Write failing image escape test**

Authorize `tmp_path / "album"`, create images inside and beside it, then call the real `/api/image` endpoint through FastAPI TestClient. The inside image returns 200; the sibling returns 403 without its absolute path in the response.

- [ ] **Step 2: Verify Red and implement canonical root check**

Use `Path.resolve(strict=True)` plus `Path.is_relative_to` against resolved authorized roots. Register a root only after `POST /api/scan` validates a folder or selected files.

- [ ] **Step 3: Add vertical tests for trash, restore, system-trash, symlink escape**

For each endpoint, add one failing test, implement the smallest root/manifest validation, and rerun before moving to the next. Never invoke Finder/system trash in tests; inject or patch that boundary.

- [ ] **Step 4: Run backend contract tests and commit**

Commit: `fix: confine photo operations to selected roots`

### Task 2: Enforce loopback browser boundary

**Files:**
- Modify: `tests/test_backend_security.py`
- Modify: `backend/main.py`
- Modify: `run.sh`
- Modify: `README.md`

- [ ] **Step 1: Write failing disallowed-origin/Host tests**

Assert an evil origin and non-loopback Host cannot reach `/api/health`; assert documented 5173 and same-origin loopback requests work.

- [ ] **Step 2: Implement strict middleware and verify Green**

Configure exact loopback CORS origins and Starlette `TrustedHostMiddleware` for `127.0.0.1`, `localhost`, and `[::1]`. Keep route tests' `testserver` host explicitly allowed only in test configuration.

- [ ] **Step 3: Replace every documented `0.0.0.0` launch**

Change `run.sh` and Windows README commands to `127.0.0.1`. Add a static regression test that rejects `--host 0.0.0.0` in launch docs/scripts.

- [ ] **Step 4: Commit**

Commit: `fix: keep album service on loopback`

### Task 3: Load checkpoints safely

**Files:**
- Create: `tests/test_checkpoint_security.py`
- Modify: `src/predict_face.py`
- Modify: `src/evaluate.py`
- Modify: `src/train_mobilenet.py`

- [ ] **Step 1: Write failing representative-checkpoint test**

Create a temporary checkpoint containing `state_dict`, `classes`, `decision_threshold`, and `arch`. Patch `torch.load` with a spy and assert `_load_model` requests `weights_only=True`.

- [ ] **Step 2: Verify Red, change one load site, verify Green**

Update `predict_face.py`, run its test, then repeat separate Red/Green slices for evaluate and training resume/best-checkpoint loads.

- [ ] **Step 3: Add malicious-pickle regression**

Create a pickle payload with a side-effect reducer in a temp directory and prove the project loader rejects it without creating the marker file.

- [ ] **Step 4: Commit**

Commit: `fix: load model checkpoints in weights-only mode`

### Task 4: Migrate to MediaPipe Tasks

**Files:**
- Create: `src/face_detector.py`
- Create: `tests/test_face_detector.py`
- Create: `scripts/download_models.py`
- Modify: `src/predict_face.py`
- Modify: `src/prepare_dataset.py`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Produces: `FaceDetector.detect(rgb: np.ndarray) -> list[FaceBox]`, where `FaceBox` contains absolute `x, y, width, height`.

- [ ] **Step 1: Write failing adapter contract with fake Tasks result**

Verify the adapter converts an absolute MediaPipe Tasks bounding box to `FaceBox` and returns an empty list for no detections.

- [ ] **Step 2: Implement adapter and verify Green**

Use `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)` and `mp.tasks.vision.FaceDetector.create_from_options` with the short-range model, image running mode, and configured confidence.

- [ ] **Step 3: Migrate prediction consumer with one failing crop test**

Inject a fake adapter returning known boxes; assert `_detect_largest_face` picks the largest eligible face and preserves `_expand_box` coordinates. Remove `mp.solutions` only after Green.

- [ ] **Step 4: Migrate dataset consumer with one failing output test**

Use a synthetic image and fake adapter; assert output count/name/224x224 size stay stable.

- [ ] **Step 5: Add checksum-verified model downloader**

Download the official short-range asset to `models/blaze_face_short_range.tflite` and verify SHA-256 `b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f`; use atomic temporary-file replacement and reject a mismatch.

- [ ] **Step 6: Update dependencies and run real boundary smoke test**

Pin compatible audited MediaPipe Tasks, remove `protobuf<4`, download the verified asset, and run detection against a generated blank image to prove the native boundary initializes and returns no faces.

- [ ] **Step 7: Commit**

Commit: `fix: migrate face detection to MediaPipe Tasks`

### Task 5: Upgrade frontend dependencies and verify the portfolio workflow

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Modify: `README.md`

- [ ] **Step 1: Capture the current frontend build as the behavior baseline**

Run `npm ci` and `npm run build` before changing versions; record exit status.

- [ ] **Step 2: Upgrade one dependency family**

Upgrade Vite and `@vitejs/plugin-vue` together to compatible audited releases, regenerate the lockfile, run build, then run `npm audit --audit-level=high`.

- [ ] **Step 3: Run full verification**

Run Python tests, Ruff, Vue build, npm audit, pip-audit, Bandit, secret scan, real MediaPipe initialization, and the temp-fixture flow scan → preview → app Trash → restore.

- [ ] **Step 4: Document security design and commit**

Document on-device processing, loopback binding, root confinement, safe checkpoint loading, model checksum verification, dependency verification, and exact commands.

Commit: `docs: explain local photo security controls`
