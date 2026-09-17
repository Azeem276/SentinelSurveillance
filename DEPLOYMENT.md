# Sentinel — Server Deployment Checklist

Follow this in order. Steps marked **REQUIRED** will break the app if skipped.

---

## 0. Before you copy the project

Do **not** copy these to the server:

| Path | Why |
|---|---|
| `.env` | Contains the development database URL. Recreate from `.env.example`. |
| `.venv/` | Platform-specific. Rebuild on the server. |
| `.devdb/` | Throwaway local PostgreSQL cluster used during development. Delete it. |
| `frontend/node_modules/` | Rebuild with `npm install`. |
| `storage/` | Local recordings, face crops and snapshots. |
| `models/` | Optional — they re-download automatically (~44 MB). |
| `data/sample_videos/` | Local test footage. |

All of these are already in `.gitignore`.

**Do** copy: `backend/`, `frontend/src` + config files, `scripts/`, `docker/`,
`.env.example`, `docker-compose*.yml`, `README.md`, this file.

---

## 1. System packages

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip \
                    postgresql-16 postgresql-client-16 \
                    nodejs npm nginx \
                    libgl1 libglib2.0-0
```

`libgl1` and `libglib2.0-0` are needed by OpenCV on headless servers —
missing them is the single most common deployment failure
(`ImportError: libGL.so.1`). If you prefer, use `opencv-python-headless`
instead; the code does not use any GUI functions.

---

## 2. Python environment — **REQUIRED**

```bash
cd /opt/sentinel
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

**GPU:** on Linux with NVIDIA, the default PyPI `torch` wheel already
includes CUDA — nothing extra. Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. PostgreSQL — **REQUIRED**

```bash
sudo -u postgres psql
```

```sql
CREATE ROLE sentinel WITH LOGIN PASSWORD 'CHANGE-ME-STRONG';
CREATE DATABASE sentinel WITH OWNER sentinel ENCODING 'UTF8';
\q
```

Apply the schema (tables, constraints, indexes, grants, seed rows):

```bash
psql -U postgres -d sentinel -f backend/migrations/schema.sql
```

Verify — expect **12** tables:

```bash
psql -U sentinel -d sentinel -c "\dt" | tail -n +4 | wc -l
psql -U sentinel -d sentinel -c "SELECT key, value FROM system_settings;"
```

The schema is idempotent (`CREATE TABLE IF NOT EXISTS`, `ON CONFLICT DO
NOTHING`), so re-running it is safe.

---

## 4. Environment file — **REQUIRED**

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env`:

```env
ENVIRONMENT=production
DEBUG=false
LOG_JSON=true

DATABASE_URL=postgresql+psycopg://sentinel:CHANGE-ME-STRONG@localhost:5432/sentinel

# python -c "import secrets; print(secrets.token_urlsafe(48))"
SECRET_KEY=<generated>

# The browser origin(s) that will load the dashboard.
CORS_ORIGINS=https://sentinel.your-domain.example

API_HOST=127.0.0.1
API_PORT=8600

AI_DEVICE=auto
AUTO_DOWNLOAD_MODELS=true

# Absolute paths are recommended in production.
STORAGE_PATH=/var/lib/sentinel/storage
VIDEO_STORAGE_PATH=/var/lib/sentinel/videos
FACE_DATASET_PATH=/var/lib/sentinel/familiar_faces
MODEL_PATH=/opt/sentinel/models
```

> Relative paths resolve against the repository root, not the working
> directory — safe either way, but absolute is clearer for a service.

Create the directories:

```bash
sudo mkdir -p /var/lib/sentinel/{storage,videos,familiar_faces}
sudo chown -R sentinel:sentinel /var/lib/sentinel
```

---

## 5. Models

```bash
source .venv/bin/activate
python scripts/download_models.py
python scripts/download_models.py --list      # confirm all three installed
```

Do this at deploy time rather than letting the first request trigger it.
For an air-gapped server, copy `yolo11n.pt`,
`face_detection_yunet_2023mar.onnx` and
`face_recognition_sface_2021dec.onnx` into `MODEL_PATH` and set
`AUTO_DOWNLOAD_MODELS=false`.

---

## 6. Smoke test before wiring up services

```bash
cd backend
../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8600
```

In another shell:

```bash
curl -s localhost:8600/api/system/health     # {"status":"ok", "database":{"connected":true}}
curl -s localhost:8600/api/system/state
curl -s localhost:8600/api/system/models
```

If `status` is `degraded`, the database is unreachable — fix step 3/4 before
continuing. Then stop it with Ctrl-C.

---

## 7. Frontend build

```bash
cd frontend
npm ci            # or: npm install
npm run build     # emits frontend/dist/
```

If the API is served from the same origin via nginx (recommended), no build
variables are needed. Otherwise set them at build time:

```bash
VITE_API_BASE=https://api.your-domain.example \
VITE_WS_URL=wss://api.your-domain.example/ws \
npm run build
```

---

## 8. systemd service

`/etc/systemd/system/sentinel.service`:

```ini
[Unit]
Description=Sentinel surveillance backend
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=sentinel
Group=sentinel
WorkingDirectory=/opt/sentinel/backend
Environment="PYTHONUNBUFFERED=1"
ExecStart=/opt/sentinel/.venv/bin/python -m uvicorn app.main:app \
          --host 127.0.0.1 --port 8600 --workers 1
Restart=always
RestartSec=5

# Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=/var/lib/sentinel /opt/sentinel/models

[Install]
WantedBy=multi-user.target
```

> **`--workers 1` is deliberate.** The surveillance manager, capture threads,
> alarm state machine and recognition index are per-process singletons.
> Multiple uvicorn workers would each start their own capture threads and
> record the same source several times. Scale by adding sources/hardware, not
> workers.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sentinel
sudo systemctl status sentinel
journalctl -u sentinel -f
```

---

## 9. nginx reverse proxy

`/etc/nginx/sites-available/sentinel`:

```nginx
server {
    listen 443 ssl http2;
    server_name sentinel.your-domain.example;

    ssl_certificate     /etc/letsencrypt/live/sentinel.your-domain.example/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/sentinel.your-domain.example/privkey.pem;

    # --- REQUIRED: no authentication is built in. Protect it here. ---
    auth_basic           "Sentinel";
    auth_basic_user_file /etc/nginx/.htpasswd;

    root /opt/sentinel/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8600;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # MJPEG streams and video playback must not be buffered.
        proxy_buffering off;
        proxy_read_timeout 3600s;
        client_max_body_size 2G;      # video uploads
    }

    location /ws {
        proxy_pass http://127.0.0.1:8600;
        proxy_http_version 1.1;
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host       $host;
        proxy_read_timeout 3600s;
    }
}

server {
    listen 80;
    server_name sentinel.your-domain.example;
    return 301 https://$host$request_uri;
}
```

```bash
sudo ln -s /etc/nginx/sites-available/sentinel /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

`proxy_buffering off` matters: with it on, the MJPEG preview never renders.

---

## 10. Post-deploy verification

```bash
curl -s https://sentinel.your-domain.example/api/system/health
```

In the browser:

1. Dashboard loads; header shows **LIVE** (WebSocket connected).
2. Add a source, or drop footage into `VIDEO_STORAGE_PATH`.
3. **ACTIVATE SURVEILLANCE** → source goes `AVAILABLE`, `REC` appears.
4. Check a recording file was created under `storage/recordings/<source>/`.
5. **ENABLE INTELLIGENCE** → Diagnostics shows non-zero processed frames and
   a real inference latency.
6. Confirm the device on Diagnostics is `cuda` if you expected a GPU.
7. Walk a known person through the scene; confirm recognition and, at
   Proximity B, the expected alert behaviour.
8. Trigger an unfamiliar alarm and confirm **STOP ALARM** works.

---

## 11. Operations

**Storage.** Recording is the dominant cost: roughly
`fps × resolution` — a 960×540 @ 15 fps `mp4v` stream is about
**0.6–1.2 GB per source per day**. Plan a retention job:

```bash
# Delete recordings older than 30 days (metadata rows remain).
find /var/lib/sentinel/storage/recordings -name '*.mp4' -mtime +30 -delete
```

**Backups.** Back up the database (metadata, identities, embeddings, events)
and `storage/faces/` (enrolled crops). Recordings are usually too large to
back up wholesale.

```bash
pg_dump -U sentinel -Fc sentinel > sentinel-$(date +%F).dump
```

**Logs.** Set `LOG_JSON=true` in production. Stable event keys worth
alerting on: `SOURCE_UNAVAILABLE`, `RECORDING_INTERRUPTED`,
`DETECTION_ERROR`, `FACE_RECOGNITION_ERROR`, `ALARM_STARTED`,
`MODEL_MISSING`, `database_unavailable`.

**Restart safety.** On boot the app closes recordings/tracks/events left
open by an unclean shutdown, marks those recordings `INTERRUPTED`, rebuilds
the recognition index, starts the expiration scheduler and restores the
operator's persisted surveillance state. Nothing is appended to an
interrupted file — a resumed recording always opens a new one.

**Temporary identity expiry** runs server-side on an APScheduler job every
`EXPIRATION_SCAN_SECONDS` (default 60). It does not depend on a browser
being open. Historical events survive expiry.

---

## 12. Security must-dos

- [ ] **Put authentication in front of it.** There is no auth layer in the
      app. nginx basic auth, an SSO proxy, or a private network — pick one.
- [ ] Never expose port 8600 publicly; bind the backend to `127.0.0.1`.
- [ ] Set a real `SECRET_KEY` and a strong database password.
- [ ] `chmod 600 .env`.
- [ ] Set `CORS_ORIGINS` to your actual origin, not `*`.
- [ ] Run as a dedicated non-root `sentinel` user.
- [ ] TLS everywhere — face imagery and recordings are sensitive data.
- [ ] Consider disk encryption for `/var/lib/sentinel`.
- [ ] Check your local law on biometric data and recording consent before
      operating this on real people.

---

## 13. Cleaning up the development machine

The development session created a throwaway PostgreSQL cluster on port
55432, entirely separate from your normal instance. To remove it:

```powershell
# Windows PowerShell, from the project root
& "C:\Program Files\PostgreSQL\18\bin\pg_ctl.exe" -D .devdb\data stop
Remove-Item -Recurse -Force .devdb
```

Your own PostgreSQL service on port 5432 was never modified.
