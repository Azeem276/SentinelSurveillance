# Sentinel

**Intelligent Video Surveillance & Analysis Platform** — a privacy-focused, locally
deployable surveillance system with real object detection, tracking, face
recognition, two-zone proximity security, recording and event analysis.

Every model runs on your own hardware. No video, face crop, embedding or
recording ever leaves the machine.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
  - [1. Clone and create a virtualenv](#1-clone-and-create-a-virtualenv)
  - [2. PostgreSQL setup](#2-postgresql-setup)
  - [3. Environment configuration](#3-environment-configuration)
  - [4. AI model installation](#4-ai-model-installation)
  - [5. Frontend setup](#5-frontend-setup)
- [Running](#running)
- [Sample video setup](#sample-video-setup)
- [Familiar face setup](#familiar-face-setup)
- [Demo data](#demo-data)
- [Running the tests](#running-the-tests)
- [GPU setup](#gpu-setup)
- [Deployment](#deployment)
- [Docker](#docker)
- [API overview](#api-overview)
- [Configuration reference](#configuration-reference)
- [Privacy and local-first guarantee](#privacy-and-local-first-guarantee)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Future CCTV / RTSP integration](#future-cctv--rtsp-integration)
- [Model licences](#model-licences)

---

## What it does

Sentinel watches a set of video sources and turns them into structured
security intelligence:

| Capability | Detail |
|---|---|
| **Object detection** | YOLO11 — person, car, truck, motorcycle, bicycle, bus, dog, cat, and other COCO classes |
| **Tracking** | Persistent `track_id` per object, surviving brief occlusion |
| **Motion detection** | MOG2 with start/end hysteresis, so one walk-past is one event |
| **Face detection** | YuNet, multiple faces at once, only for people inside the recognition zone |
| **Face recognition** | SFace 128-d embeddings, cosine similarity, configurable threshold |
| **Two-zone proximity** | Proximity A (recognition) and Proximity B (alarm), per source |
| **Security rules** | A dedicated rule engine, deliberately separate from the AI models |
| **Recording** | One file per uninterrupted session, safe close, never appends to an interrupted file |
| **Events** | Structured, deduplicated, with durations — not one row per frame |
| **Alerts** | Backend-owned alarm state machine, continuous alarm until manually stopped |

### The state distinctions that matter

Sentinel keeps these strictly separate, because conflating them is how
surveillance systems produce false alarms:

```
DETECTED  ≠  FACE DETECTED  ≠  FACE RECOGNIZED  ≠  FAMILIAR  ≠  UNFAMILIAR  ≠  ALARM
TRACK ID  ≠  IDENTITY ID
PROXIMITY A (recognition, farther)  ≠  PROXIMITY B (alarm, closer)
```

Recognition states are explicit:

| State | Meaning | Alarms at Proximity B? |
|---|---|---|
| `NO_FACE` | Person tracked, no face found | No |
| `UNKNOWN_PENDING_RECOGNITION` | Too far to attempt recognition yet | **No** |
| `FACE_UNRECOGNIZABLE` | Face found but too small/blurry/angled to evaluate | **No** (configurable) |
| `UNFAMILIAR` | Face *was* evaluated and matched nobody | **Yes — continuous alarm** |
| `TEMPORARY_FAMILIAR` | Matched a temporary identity | One beep (configurable) |
| `PERMANENT_FAMILIAR` | Matched a permanent identity | Never |

Only `UNFAMILIAR` justifies an intruder alarm: a *failure* to recognise
someone is not evidence of an intruder.

---

## Architecture

```
                            ┌──────────────────────┐
                            │   React + TS (Vite)  │
                            │  operator dashboard  │
                            └───────┬───────┬──────┘
                       REST / MJPEG │       │ WebSocket (structured state)
                            ┌───────┴───────┴──────┐
                            │   FastAPI  (app.api) │
                            └──────────┬───────────┘
                            ┌──────────┴───────────┐
                            │ Application services │
                            │ surveillance·identity│
                            │ analysis·scheduler   │
                            └──────────┬───────────┘
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        │              │               │               │              │
┌───────┴──────┐ ┌─────┴──────┐ ┌──────┴───────┐ ┌─────┴─────┐ ┌──────┴──────┐
│ Video source │ │ Recording  │ │ Intelligence │ │  Events   │ │   Alerts    │
│  abstraction │ │   engine   │ │   pipeline   │ │  engine   │ │   engine    │
│ File │ RTSP  │ │ 1 file per │ │              │ │ dedup +   │ │ alarm state │
│              │ │  session   │ │              │ │ aggregate │ │  machine    │
└──────────────┘ └────────────┘ └──────┬───────┘ └───────────┘ └─────────────┘
                                       │
          ┌──────────┬─────────┬───────┼────────┬──────────┬──────────┐
     ┌────┴────┐ ┌───┴────┐ ┌──┴───┐ ┌─┴─────┐ ┌┴────────┐ ┌┴────────┐
     │Detection│ │Tracking│ │Motion│ │ Face  │ │Proximity│ │Security │
     │ (YOLO)  │ │ByteTrk │ │ MOG2 │ │Yu/SF  │ │estimator│ │  rules  │
     └─────────┘ └────────┘ └──────┘ └───────┘ └─────────┘ └─────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │  Repositories → PostgreSQL (metadata)│
                    │  Storage layer → filesystem (media)  │
                    └─────────────────────────────────────┘
```

Per frame, the pipeline runs:

```
motion → detection → tracking → proximity
       → (person AND inside Proximity A) face detection
       → face quality gate → embedding → recognition
       → identity classification → proximity events
       → security rules → alerts
```

### Project layout

```
sentinel/
├── backend/
│   ├── app/
│   │   ├── api/            REST routers + WebSocket
│   │   ├── core/           config, logging, hardware detection, exceptions
│   │   ├── db/             engine/session management
│   │   ├── models/         SQLAlchemy entities
│   │   ├── schemas/        Pydantic request/response models
│   │   ├── repositories/   all query construction lives here
│   │   ├── services/       surveillance manager, workers, identity, analysis
│   │   ├── video/          VideoSource abstraction (File, RTSP)
│   │   ├── recording/      recording engine
│   │   ├── intelligence/   detection, tracking, motion, face, proximity
│   │   ├── events/         bus + dedup/aggregation engine
│   │   ├── alerts/         alarm state machine
│   │   ├── security/       security rule engine (pure functions)
│   │   └── storage/        path safety
│   ├── migrations/schema.sql   ← the authoritative SQL schema
│   ├── tests/
│   └── requirements.txt
├── frontend/src/{components,pages,hooks,services,stores,types,styles}
├── data/{sample_videos,familiar_faces}
├── storage/{recordings,faces,snapshots,events}
├── models/                 downloaded model weights
├── scripts/                setup, schema generation, demo data
├── docker/ + docker-compose.yml
└── .env.example
```

---

## Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.11+ | Tested on 3.13 |
| PostgreSQL | 14+ | Tested on 18.1 |
| Node.js | 20+ | For the frontend |
| RAM | 4 GB | 8 GB+ for several sources |
| Disk | ~2 GB | Plus recording storage |
| GPU | optional | CUDA used automatically if present; CPU otherwise |

FFmpeg is bundled with the `opencv-python` wheel, so a separate install is
only needed for unusual codecs.

---

## Installation

### 1. Clone and create a virtualenv

```bash
cd sentinel
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r backend/requirements.txt        # add -r backend/requirements-dev.txt for tests
```

### 2. PostgreSQL setup

The full schema is in **`backend/migrations/schema.sql`** — tables,
constraints, indexes, roles and grants. It is generated from the SQLAlchemy
models, so it never drifts from the code.

Create the role and database as a superuser (**change the password**):

```sql
CREATE ROLE sentinel WITH LOGIN PASSWORD 'your-strong-password';
CREATE DATABASE sentinel WITH OWNER sentinel ENCODING 'UTF8';
```

Then apply the schema:

```bash
psql -U postgres -d sentinel -f backend/migrations/schema.sql
```

Verify:

```bash
psql -U sentinel -d sentinel -c "\dt"          # expect 12 tables
```

<details>
<summary>Tables created</summary>

`video_sources`, `recording_sessions`, `tracks`, `detections`,
`motion_events`, `identities`, `faces`, `face_embeddings`,
`temporary_identity_expirations`, `security_events`, `alerts`,
`system_settings`

Indexed on `source_id`, `timestamp`, `track_id`, `identity_id` and
`event_type`, with partial indexes for the hot "what is happening now"
queries.
</details>

If you change a model, regenerate the schema:

```bash
python scripts/generate_schema.py
```

**Face embeddings** are stored as `BYTEA` (little-endian float32) plus a
dimension column, so **pgvector is not required**. Similarity search runs in
an in-memory index that is rebuilt whenever the identity dataset changes.
`schema.sql` section 6 shows how to switch to pgvector if you prefer.

### 3. Environment configuration

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
DATABASE_URL=postgresql+psycopg://sentinel:your-strong-password@localhost:5432/sentinel
SECRET_KEY=<python -c "import secrets; print(secrets.token_urlsafe(48))">
CORS_ORIGINS=http://localhost:5173
```

`.env` is gitignored. Relative paths in it are resolved against the
repository root, not the working directory, so the backend behaves the same
however it is launched.

### 4. AI model installation

Models download automatically on first use (`AUTO_DOWNLOAD_MODELS=true`).
To provision them explicitly — recommended for a server, so the first
request is not slow:

```bash
python scripts/download_models.py            # fetch anything missing
python scripts/download_models.py --list     # inventory + licences
python scripts/download_models.py --force    # re-download everything
```

| Model | File | Size | Licence | Purpose |
|---|---|---|---|---|
| YOLO11n | `yolo11n.pt` | 5.6 MB | AGPL-3.0 (Ultralytics) | Object detection + ByteTrack |
| YuNet | `face_detection_yunet_2023mar.onnx` | 0.2 MB | MIT (OpenCV Zoo) | Face detection |
| SFace | `face_recognition_sface_2021dec.onnx` | 38 MB | Apache-2.0 (OpenCV Zoo) | Face embeddings |

If a model is missing and auto-download is off, the API returns a clear
`Model not installed` error naming the command to run.

### 5. Frontend setup

```bash
cd frontend
npm install
```

---

## Running

Two terminals.

**Backend:**

```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8600 --reload
```

**Frontend:**

```bash
cd frontend
npm run dev
```

| | URL |
|---|---|
| **Dashboard** | **http://localhost:5173** |
| API docs (Swagger) | http://localhost:8600/api/docs |
| Health check | http://localhost:8600/api/system/health |
| WebSocket | ws://localhost:8600/ws |

The Vite dev server proxies `/api` and `/ws` to the backend, so the browser
sees a single origin.

### First run

1. Open the dashboard.
2. Press **ACTIVATE SURVEILLANCE** — sources start, recording begins.
3. Press **ENABLE INTELLIGENCE** — detection, tracking and recognition start.
4. Select a source on the left for the single view, or switch to **Grid**.

Surveillance is the parent state: intelligence cannot run without it.

---

## Sample video setup

Put any `.mp4` / `.mkv` / `.avi` / `.mov` into `data/sample_videos/`, then add
a source from **Monitor → + Add source** (or upload through that dialog).

No footage is committed to this repository. Two generators are provided:

```bash
# Synthetic clips: exercise capture, recording, motion, streaming.
# YOLO finds little in drawn shapes - that is expected.
python scripts/make_sample_videos.py

# A clip built from a real photograph: produces genuine detections,
# real track ids and real Proximity A/B transitions.
python scripts/make_detection_clip.py
python scripts/make_detection_clip.py --photo /path/to/your/photo.jpg
```

For the full pipeline including face recognition, use your own footage where
faces are reasonably large, lit and front-facing.

---

## Familiar face setup

Enrol known people from disk:

```
data/familiar_faces/
├── permanent/
│   ├── Azeem/       img1.jpg  img2.jpg  img3.jpg
│   └── Sarah Khan/  img1.jpg
└── temporary/
    └── John Smith/  img1.jpg  img2.jpg
```

Use several photos per person, varying angle and lighting. Then:

```bash
curl -X POST http://localhost:8600/api/identities/enrol-dataset
```

or press **Enrol dataset folder** on the Identities page. The directory name
becomes the display name, and the recognition index rebuilds immediately.

You can also enrol from live detections: unknown faces appear on the
**Review** page, where you classify them as permanent or temporary, with an
optional name.

---

## Demo data

To explore the whole UI without waiting for real footage:

```bash
python scripts/seed_demo_data.py --reset
```

This writes through the real repositories, so the shape of the data matches
what the live pipeline produces: 3 sources, 5 identities (permanent,
temporary and one already-expired), recording sessions (completed and
interrupted), tracks, detections, motion episodes, ~95 events, alarms
including one left **ACTIVE** so you can exercise **STOP ALARM**, and
unfamiliar faces pending review.

The face images it stores are obviously synthetic placeholders — they exist
to exercise the review workflow, not to imitate real biometric samples.

---

## Running the tests

```bash
pip install -r backend/requirements-dev.txt
cd backend
python -m pytest -q                    # whole suite
python -m pytest tests/test_proximity.py -v
python -m pytest --cov=app --cov-report=term-missing
```

The suite runs against SQLite and small synthetic media, so it needs no
database server and no large fixtures. Tests requiring the real model
weights skip cleanly when they are not installed.

Coverage includes: proximity zones and the A > B invariant, the full
approach/departure sequence, security rules for every recognition state,
track-level identity memory and temporal consistency, identity creation,
naming, timestamp-identifier fallback, bulk classification, expiration,
index synchronisation, recording lifecycle (start/stop/restart/interrupt),
event deduplication, the alarm state machine, video source abstraction,
path-traversal defences, tracking and motion detection.

Frontend type checking:

```bash
cd frontend && npm run lint
```

---

## GPU setup

Hardware is detected at startup and CUDA is **never** mandatory.

```env
AI_DEVICE=auto     # auto | cuda | cpu | mps
```

`auto` uses CUDA when available, then MPS, then CPU. Requesting `cuda` on a
machine without it logs the reason and falls back to CPU rather than
crashing.

On **Linux** with an NVIDIA GPU, the default PyPI `torch` wheel already
includes CUDA — nothing extra to do. On **Windows**, the default wheel is
CPU-only; install a CUDA build explicitly:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Confirm what is in use:

```bash
curl http://localhost:8600/api/system/diagnostics | python -m json.tool
```

The **Diagnostics** page reports the active device, per-source FPS,
inference/detection/face latency, dropped frames, active tracks, memory and
CPU — measured, not estimated.

---

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the server checklist: production
`.env`, systemd units, building the frontend, reverse proxy and TLS,
storage/backup planning, and hardening notes.

Quick production build of the frontend:

```bash
cd frontend
npm run build          # emits frontend/dist/ for any static host
```

Set `VITE_API_BASE` and `VITE_WS_URL` at build time if the API is on a
different origin.

---

## Docker

```bash
cp .env.example .env          # set POSTGRES_PASSWORD and DATABASE_URL
docker compose up -d
```

Brings up PostgreSQL (schema applied automatically on first boot), the
backend and the frontend.

GPU inference inside Docker needs the NVIDIA Container Toolkit; the
`docker-compose.gpu.yml` overlay is provided:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

For local GPU development, running the backend natively and only
PostgreSQL in Docker is usually simpler:

```bash
docker compose up -d postgres
```

---

## API overview

Full interactive docs at `/api/docs`.

**System**
```
GET  /api/system/health          GET  /api/system/state
GET  /api/system/diagnostics     GET  /api/system/models
GET  /api/system/config
```

**Surveillance & intelligence**
```
POST /api/surveillance/start     POST /api/surveillance/stop
POST /api/intelligence/start     POST /api/intelligence/stop
POST /api/sources/{id}/intelligence/start
POST /api/sources/{id}/intelligence/stop
```

**Sources**
```
GET    /api/sources              POST   /api/sources
GET    /api/sources/{id}         PATCH  /api/sources/{id}
DELETE /api/sources/{id}
POST   /api/sources/{id}/start   POST   /api/sources/{id}/stop
GET    /api/sources/{id}/stream      (MJPEG live preview)
GET    /api/sources/{id}/snapshot
GET    /api/sources/{id}/events      /detections  /tracks  /motion  /analysis
POST   /api/sources/upload
```

**Identities & faces**
```
GET    /api/identities           POST   /api/identities
PATCH  /api/identities/{id}      DELETE /api/identities/{id}
POST   /api/identities/enrol-dataset
POST   /api/identities/sync-index
POST   /api/identities/expire-now
GET    /api/faces/unfamiliar
POST   /api/faces/{id}/classify
POST   /api/faces/classify-bulk
GET    /api/faces/image/{path}
```

**Events, alerts, archive**
```
GET  /api/events          GET  /api/events/recent
GET  /api/alerts          GET  /api/alerts/active
POST /api/alerts/{id}/stop        POST /api/alerts/stop-all
GET  /api/security/active
GET  /api/recordings      GET  /api/recordings/{id}
GET  /api/recordings/{id}/play    (HTTP range, seekable)
GET  /api/archive/tree
```

**WebSocket** `ws://host/ws?topics=…&sources=…` pushes
`frame_analysis`, `motion`, `event`, `recognition`, `proximity`,
`track_started`, `track_ended`, `alarm_started`, `alarm_stopped`,
`alert_beep`, `source_state` and `system_state`.

Video pixels go over MJPEG, not the WebSocket, so the realtime channel stays
small and overlays render as crisp vectors client-side.

---

## Configuration reference

Everything is read through one settings class — no scattered `os.environ`
lookups. Full list in `.env.example`; the ones you are most likely to tune:

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection (psycopg 3 driver) |
| `AI_DEVICE` | `auto` | `auto` / `cuda` / `cpu` / `mps` |
| `FACE_RECOGNITION_THRESHOLD` | `0.55` | Cosine similarity to accept a match |
| `DEFAULT_PROXIMITY_A` | `10.0` | Recognition zone, metres (**must exceed B**) |
| `DEFAULT_PROXIMITY_B` | `3.0` | Alarm zone, metres |
| `TEMPORARY_FAMILIAR_DEFAULT_DAYS` | `7` | Default temporary retention |
| `DETECTION_INTERVAL` | `3` | Run the detector every N frames |
| `TARGET_FPS` | `12` | Analysis cadence per source |
| `FRAME_MAX_WIDTH` | `960` | Downscale before inference |
| `RECOGNITION_COOLDOWN_FRAMES` | `45` | Frames before re-verifying an identity |
| `ALARM_UNRECOGNIZABLE_POLICY` | `ignore` | `ignore` / `alarm` for unreadable faces |
| `TEMPORARY_FAMILIAR_ALERT` | `beep` | `none` / `beep` / `continuous` |
| `RECORDING_MAX_MINUTES` | `30` | Rotate to a new file (0 disables) |

`DEFAULT_PROXIMITY_A > DEFAULT_PROXIMITY_B > 0` is validated at startup and
the app refuses to boot with an inverted configuration. The same rule is
enforced per source by a database `CHECK` constraint, the API schema and the
settings UI.

### Per-source settings

Proximity A/B, recognition threshold, detection confidence, temporary
retention, alert policy and camera calibration are all configurable per
source, because a front door and a back gate have different geometry:

```
Front Door:  A = 10 m, B = 3 m
Back Gate:   A = 15 m, B = 5 m
```

Changes apply to a running source immediately — no restart.

---

## Privacy and local-first guarantee

- **All inference is local.** YOLO, YuNet and SFace run in-process on your
  CPU/GPU.
- **No cloud AI.** No video, frame, face crop, embedding or recording is sent
  to any external service. The only outbound network traffic is the one-time
  model download from GitHub, which you can do offline by copying the three
  files into `models/`.
- **Media stays on disk.** PostgreSQL holds metadata and paths only; large
  media never goes into the database.
- **Face crops are written sparingly** — only when actionable (an unfamiliar
  face awaiting review, or an enrolled identity), not every frame.
- **Logs avoid biometric content.** Structured logs carry ids, states and
  timings, not face data.

---

## Security

- All API input validated with Pydantic, including cross-field rules.
- **Path traversal is blocked at a single choke point** (`app/storage/paths.py`):
  every client- or database-supplied path is resolved inside a configured
  root; absolute paths, drive letters, UNC paths and `..` are rejected.
- Uploaded video types are allow-listed and filenames sanitised.
- Filesystem paths are not exposed by read APIs.
- All database access goes through SQLAlchemy with bound parameters.
- CORS is configurable and defaults to localhost only.
- Secrets come from the environment; no credentials are committed.

There is **no authentication layer yet** — do not expose Sentinel directly to
the internet. Put it behind a reverse proxy with authentication, or keep it
on a trusted network. See DEPLOYMENT.md.

---

## Troubleshooting

**`Cannot reach the Sentinel backend`**
Backend is not running or is on another port. Check
`curl http://localhost:8600/api/system/health`.

**Backend starts but logs `database_unavailable`**
`DATABASE_URL` is wrong or the schema has not been applied. The API still
starts so you can see the error on `/api/system/health`.

**A source shows `UNAVAILABLE`**
The file is missing from `VIDEO_STORAGE_PATH`, or the codec cannot be
decoded. `GET /api/sources/{id}` returns the exact error in `last_error`.
Relative paths in `.env` resolve against the repository root.

**No detections on the synthetic clips**
Expected. YOLO is trained on photographic imagery. Use
`scripts/make_detection_clip.py` or your own footage.

**Faces detected but never recognised**
Faces are probably failing the quality gate. Check `quality_reason` on
`/api/faces/unfamiliar` (`face_too_small`, `too_blurry`, `too_dark`,
`extreme_pose`). Lower `FACE_MIN_PIXELS`, or move the camera closer. This is
working as intended — it reports `FACE_UNRECOGNIZABLE` instead of guessing.

**A known person is not matched**
Lower `FACE_RECOGNITION_THRESHOLD` (try 0.45) or enrol more photos of them.
Rebuild with `POST /api/identities/sync-index`.

**Alarm will not stop**
By design a continuous alarm runs until acknowledged. Use **STOP ALARM**, or
`POST /api/alerts/{id}/stop`, or `POST /api/alerts/stop-all`.

**No sound**
Browsers block audio before a user gesture. Click anywhere, and check the
mute button in the header.

**Low FPS on CPU**
Raise `DETECTION_INTERVAL`, lower `TARGET_FPS` or `FRAME_MAX_WIDTH`. Check
real numbers on the Diagnostics page.

**Recording files are 0 bytes / will not open**
The `mp4v` codec is missing. Try `RECORDING_FOURCC=XVID` with `.avi`, or
install a fuller FFmpeg.

**`psycopg` install fails**
Use the binary wheel: `pip install "psycopg[binary]"`.

---

## Future CCTV / RTSP integration

The MVP uses files, but nothing above the video layer knows that. Every
source is a `VideoSourceAdapter` exposing `open()`, `read() -> Frame` and
`release()`, and the intelligence engine only ever sees `Frame`.

`RTSPCameraSource` is already implemented — TCP transport, single-frame
buffering so a live source cannot build a backlog, and reconnection with
exponential backoff. To use a camera:

```bash
curl -X POST http://localhost:8600/api/sources \
  -H 'Content-Type: application/json' \
  -d '{"uid":"front_door","name":"Front Door","type":"RTSP",
       "uri":"rtsp://user:pass@192.168.1.50:554/stream1",
       "proximity_a":10.0,"proximity_b":3.0}'
```

Nothing in detection, tracking, recognition, proximity, recording, events,
alerts or the UI changes. Adding another source kind means writing one
adapter and one branch in `build_source_adapter()`.

### Local AI / LLM integration

Sentinel is standalone and depends on no external AI system, but exposes
clean, stable, structured reads intended for a local LLM or agent:

```
GET /api/events/recent       structured event objects
GET /api/sources/{id}/analysis
GET /api/security/active     consolidated "what is happening now"
```

---

## Model licences

| Model | Licence | Source |
|---|---|---|
| YOLO11n | **AGPL-3.0** | https://github.com/ultralytics/assets |
| YuNet | MIT | https://github.com/opencv/opencv_zoo |
| SFace | Apache-2.0 | https://github.com/opencv/opencv_zoo |

**Note on YOLO:** Ultralytics is AGPL-3.0. That is fine for internal,
research and personal use, but distributing a product built on it requires
either releasing your source under AGPL-3.0 or buying an Ultralytics
commercial licence. The detector sits behind the `ObjectDetector` interface
(`app/intelligence/detection/base.py`), so swapping in a permissively
licensed model means implementing one class — nothing else changes.
