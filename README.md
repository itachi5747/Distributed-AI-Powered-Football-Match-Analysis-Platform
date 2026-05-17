<div align="center">

# ⚽ FootballIQ

### Distributed AI-Powered Football Match Analysis Platform

*Automatically extract 30+ professional-grade statistics from any football match video*

<br/>

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-3.13-FF6600?style=for-the-badge&logo=rabbitmq&logoColor=white)](https://rabbitmq.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

<br/>

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)](.)
[![Phase](https://img.shields.io/badge/Phase-3%20of%206-blue?style=flat-square)](.)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](.)

<br/>

</div>

---

## 📖 What Is FootballIQ?

FootballIQ is a **production-grade microservice backend** that transforms raw football match footage into deep, actionable analytics — automatically. Upload a match video (or connect a live camera stream), and FootballIQ's computer vision pipeline will detect every player, track their movement, and compute elite-level statistics in real time.

No manual tagging. No human annotation. Just upload and analyze.



---

## ✨ Key Features

| Feature | Status | Description |
|---|---|---|
| 🔐 JWT Authentication | ✅ Live | Secure register/login, refresh tokens, API keys |
| 📤 Video Upload | ✅ Live | Chunked upload to MinIO, metadata extraction |
| 🎯 Player Detection | ✅ Live | YOLOv8x — detects players, ball, referees |
| 🏃 Player Tracking | ✅ Live | ByteTrack multi-object tracking across frames |
| 📊 30+ Statistics | 🔜 Phase 4 | Possession, speed, xG, passes, formations... |
| 📡 Live WebSockets | 🔜 Phase 5 | Stream stats in real-time during live matches |
| 📄 PDF Reports | 🔜 Phase 6 | Auto-generated professional match reports |
| 💬 AI Commentary | 🔜 Phase 6 | GPT-4 powered narrative match summaries |

---

## 🏗 Architecture

```
                    ┌─────────────────────────────────┐
                    │           CLIENTS                │
                    │  (Web App / Mobile / API Users)  │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │      FastAPI Gateway (8000)      │
                    │  Auth · Matches · Stats · WS     │
                    └──┬───────┬───────┬──────────┬──┘
                       │       │       │          │
            ┌──────────┘   ┌───┘   ┌───┘      ┌───┘
            ▼              ▼       ▼           ▼
      ┌──────────┐  ┌──────────┐ ┌────────┐ ┌────────┐
      │PostgreSQL│  │  Redis   │ │RabbitMQ│ │ MinIO  │
      │    16    │  │    7     │ │  3.13  │ │  S3    │
      │(pgvector)│  │(cache,   │ │(queues)│ │(videos,│
      │          │  │sessions) │ │        │ │reports)│
      └──────────┘  └──────────┘ └───┬────┘ └────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                   ▼
          ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
          │  Inference   │  │    Stats     │  │   Report     │
          │   Worker     │  │   Engine     │  │   Worker     │
          │YOLOv8+ByteTrack│ │NumPy Engine │  │PDF Generator │
          └──────────────┘  └──────────────┘  └──────────────┘
```

---

## 📁 Repository Structure

```
footballiq/
│
├── 🐳 docker-compose.yml         # Full dev environment (DB, Redis, MQ, MinIO)
├── 📋 Makefile                   # Dev shortcuts (make up, make migrate, etc.)
├── ⚙️  pyproject.toml             # Shared tooling config (ruff, pytest, mypy)
├── 🔧 .env.example               # Template — copy to .env to configure locally
│
├── 📦 services/
│   ├── api/                      # ✅ FastAPI Gateway (Phases 1 & 2 complete)
│   │   ├── main.py               # App entrypoint with lifespan management
│   │   ├── requirements.txt
│   │   └── app/
│   │       ├── config.py         # Pydantic Settings (reads from .env)
│   │       ├── database.py       # Async SQLAlchemy engine & session factory
│   │       ├── models/           # SQLAlchemy ORM models
│   │       │   ├── user.py       # User + UserPlan enum
│   │       │   ├── api_key.py    # API Key management
│   │       │   └── match.py      # Match + MatchStatus enum
│   │       ├── schemas/          # Pydantic v2 request/response models
│   │       ├── routers/          # FastAPI route handlers
│   │       │   ├── auth.py       # /api/v1/auth/* (register, login, me, logout...)
│   │       │   └── matches.py    # /api/v1/matches/* (upload, list, get, delete...)
│   │       ├── services/         # Business logic layer
│   │       │   ├── auth_service.py
│   │       │   └── ingestion_service.py
│   │       └── dependencies/     # FastAPI Depends() factories
│   │           ├── auth.py       # get_current_user()
│   │           └── redis.py      # get_redis()
│   │
│   ├── inference/                # 🔜 CV Pipeline Worker (Phase 3)
│   ├── stats/                    # 🔜 Stats Engine Worker (Phase 4)
│   └── report/                   # 🔜 Report Generator (Phase 6)
│
├── 🔗 shared/                    # Shared package used by all services
│   ├── schemas.py                # DetectionObject, FrameMessage, InferenceJobMessage
│   ├── messaging.py              # RabbitMQ publish/consume helpers (aio-pika)
│   ├── storage.py                # MinIO upload/download/presign helpers
│   └── constants.py              # Queue names, Redis keys, CV thresholds
│
├── 🗃️  migrations/               # Alembic database migrations
│   ├── env.py
│   ├── alembic.ini
│   └── versions/
│       ├── 20260515_0001_create_users_and_api_keys.py
│       └── 20260515_0002_create_matches.py
│
└── 🧪 tests/
    └── conftest.py
```

---

## 🚀 Quick Start (Local Development)

### Prerequisites

- [Docker & Docker Compose](https://docs.docker.com/get-docker/)
- Python 3.11+
- `ffmpeg` (for video metadata extraction): `sudo apt install ffmpeg`

### Step 1 — Clone & Configure

```bash
git clone https://github.com/<your-username>/footballiq.git
cd footballiq

# Copy environment template and fill in your values
cp .env.example .env
```

The key variables you must set in `.env`:

```bash
# .env (critical values)
JWT_SECRET_KEY=your-super-secret-key-minimum-32-characters

# These are pre-configured for local Docker Compose, no changes needed:
DATABASE_URL=postgresql+asyncpg://postgres:changeme@localhost:5432/footballiq
REDIS_URL=redis://localhost:6379/0
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
MINIO_ENDPOINT=localhost:9000
```

### Step 2 — Start Infrastructure

```bash
# Start all services (PostgreSQL, Redis, RabbitMQ, MinIO)
docker compose up -d

# Verify everything is healthy
docker compose ps
```

### Step 3 — Set Up the API

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r services/api/requirements.txt

# Create symlinks so the API can find shared config and modules
ln -sf ../../.env services/api/.env
ln -sf ../../shared services/api/shared  # if not already present
```

### Step 4 — Run Migrations

```bash
# Apply all database migrations (from project root)
.venv/bin/python -m alembic -c migrations/alembic.ini upgrade head
```

### Step 5 — Start the API Server

```bash
cd services/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now running at **http://localhost:8000**

---

## 📡 API Endpoints

### Health Check
```
GET /health
→ { "status": "ok", "version": "1.0.0", "env": "development" }
```

### 🔐 Authentication (`/api/v1/auth`)
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/register` | ❌ | Create a new user account |
| `POST` | `/login` | ❌ | Login and receive JWT tokens |
| `GET` | `/me` | ✅ | Get the current user's profile |
| `POST` | `/refresh` | ✅ | Refresh an access token |
| `POST` | `/logout` | ✅ | Invalidate the current token |
| `GET` | `/keys` | ✅ | List all API keys |
| `POST` | `/keys` | ✅ | Create a new API key |
| `DELETE`| `/keys/{id}` | ✅ | Revoke an API key |

### 🎬 Match Management (`/api/v1/matches`)
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/upload` | ✅ | Upload a match video (triggers CV pipeline) |
| `GET` | `/` | ✅ | List all matches (paginated) |
| `GET` | `/{id}` | ✅ | Get full match details and progress |
| `GET` | `/{id}/status` | ✅ | Poll processing status |
| `PATCH`| `/{id}` | ✅ | Update match metadata |
| `DELETE`| `/{id}` | ✅ | Delete a match and its video |

> 📖 **Interactive Docs**: Visit [http://localhost:8000/docs](http://localhost:8000/docs) for full Swagger UI.

---

## 🗺️ Development Roadmap

```
Phase 0  ██████████ 100%  Infrastructure (Docker, PostgreSQL, Redis, RabbitMQ, MinIO)
Phase 1  ██████████ 100%  Authentication (JWT, API Keys, Redis blacklisting)
Phase 2  ██████████ 100%  Match Ingestion (Upload, MinIO, FFprobe, RabbitMQ dispatch)
Phase 3  ██████████ 100%  Inference Worker (YOLOv8x + ByteTrack)
Phase 4  ░░░░░░░░░░   0%  Stats Engine (Possession, xG, Heatmaps, Formations)
Phase 5  ░░░░░░░░░░   0%  Real-time WebSockets
Phase 6  ░░░░░░░░░░   0%  PDF Reports & AI Commentary
```

---

## 📊 Statistics (Phase 4 Target)

FootballIQ will compute the following analytics per match:

**Team Statistics**
- Ball Possession (%)
- Total Distance Covered (km)
- Pass Accuracy (%)
- Shots on Target / Off Target
- Expected Goals (xG)
- Pressing Intensity

**Player Statistics**
- Distance Covered & Sprint Count
- Top Speed (km/h)
- Heat Map
- Pass Network & Involvement
- Duels Won/Lost

---

## 🛠 Makefile Commands

```bash
make up          # docker compose up -d
make down        # docker compose down
make logs        # docker compose logs -f
make migrate     # Run alembic upgrade head
make lint        # ruff check . && mypy services/
make test        # pytest tests/ -v
make shell-api   # Open bash in API container
make shell-db    # Open psql in DB container
```

---

## 🤝 Contributing

This is an active development project. Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit with conventional commits (`feat:`, `fix:`, `chore:`, `docs:`)
4. Push and open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**Built with ❤️ for the beautiful game**

*FootballIQ — Where Computer Vision Meets Football Intelligence*

</div>
