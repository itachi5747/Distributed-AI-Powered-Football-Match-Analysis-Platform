# FootballIQ — AI-Powered Football Match Analysis Platform
## Product Requirements Document (PRD)

> **Version:** 2.0 — Detailed Edition  
> **Status:** Draft for Engineering Review  
> **Stack:** FastAPI · Python 3.11 · YOLOv8 · OpenCV · PostgreSQL · Redis · MinIO  
> **Author:** Engineering Team  
> **Date:** April 2025  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Goals](#2-problem-statement--goals)
3. [System Architecture](#3-system-architecture)
4. [Service 1 — Authentication & User Management](#4-service-1--authentication--user-management)
5. [Service 2 — Match Ingestion](#5-service-2--match-ingestion)
6. [Service 3 — Inference Engine (CV Pipeline)](#6-service-3--inference-engine-cv-pipeline)
7. [Service 4 — Statistics Engine](#7-service-4--statistics-engine)
8. [Service 5 — Live Streaming & WebSocket](#8-service-5--live-streaming--websocket)
9. [Service 6 — Player Analytics](#9-service-6--player-analytics)
10. [Service 7 — Expected Goals (xG) Module](#10-service-7--expected-goals-xg-module)
11. [Service 8 — AI Commentary Engine](#11-service-8--ai-commentary-engine)
12. [Service 9 — Report Generation](#12-service-9--report-generation)
13. [Service 10 — Tactical Analysis](#13-service-10--tactical-analysis)
14. [Service 11 — Match Chat Assistant](#14-service-11--match-chat-assistant)
15. [Service 12 — Highlight Extraction](#15-service-12--highlight-extraction)
16. [Service 13 — Multi-Camera Fusion](#16-service-13--multi-camera-fusion)
17. [AI/ML Models & Algorithms — Deep Dive](#17-aiml-models--algorithms--deep-dive)
18. [Database Schema — Full Detail](#18-database-schema--full-detail)
19. [API Design Conventions](#19-api-design-conventions)
20. [Error Handling & Response Formats](#20-error-handling--response-formats)
21. [Authentication & Security](#21-authentication--security)
22. [Performance & Scalability](#22-performance--scalability)
23. [Infrastructure & Deployment](#23-infrastructure--deployment)
24. [Non-Functional Requirements](#24-non-functional-requirements)
25. [Technology Stack — Full Reference](#25-technology-stack--full-reference)
26. [Implementation Roadmap](#26-implementation-roadmap)
27. [Appendix — Stat Calculation Formulas](#27-appendix--stat-calculation-formulas)

---

## 1. Executive Summary

FootballIQ is a **real-time and post-match football analytics platform** built on FastAPI. It ingests video files (MP4, MKV, AVI) or live camera streams of football matches and produces rich tactical and statistical insights using state-of-the-art computer vision and machine learning models.

The platform exposes a clean **REST + WebSocket API** consumable by:
- Coaching dashboards and tablet apps
- Broadcast overlays and sports data feeds
- Club performance management systems
- Amateur leagues and football academies
- Third-party sports analytics integrators

The system is designed to run on a **single NVIDIA RTX 3090** for on-premise deployment and scales horizontally on Kubernetes for cloud operations. End-to-end inference latency for a live feed is **under 200ms per frame** enabling real-time overlays and stat broadcasts at up to 30 FPS.

### What FootballIQ Produces

Given only a video of a football match, FootballIQ automatically outputs:

| Category | Statistics |
|---|---|
| **Possession** | Per-team %, time-series, zone-based possession |
| **Scoring** | Goals, shots, shots on target, shot map, xG per shot |
| **Passing** | Total passes, completion %, pass map, pass networks |
| **Physical** | Distance covered, speed, sprints, acceleration events |
| **Tactical** | Formation detection, pressing intensity (PPDA), defensive line height |
| **Events** | Goals, shots, tackles, fouls, corners, offsides, saves, dribbles |
| **Player** | Per-player stats, heatmaps, trajectory, action timeline |
| **Advanced** | Expected goals (xG), momentum index, offside line, player clusters |
| **AI** | Natural language commentary, match chat assistant, highlight reel |

---

## 2. Problem Statement & Goals

### 2.1 Problem

Professional-grade football analytics (Opta, StatsBomb, Wyscout) cost **$50,000–$500,000/year** and require dedicated data entry operators watching every match. This makes them inaccessible to:
- Football academies and youth clubs
- Amateur and semi-professional leagues
- Smaller professional clubs outside top divisions
- Broadcast companies for lower-tier matches
- Research institutions and sports scientists

### 2.2 Opportunity

Advances in open-source computer vision (YOLOv8, ByteTrack) and commodity GPU hardware now make it feasible to automate 95% of what expensive manual annotation services provide, at a fraction of the cost, running on a single server.

### 2.3 Goals

| Priority | Goal | Success Metric |
|---|---|---|
| P0 | Detect players and ball with high accuracy | mAP@0.5 > 0.88 on test set |
| P0 | Classify players into teams automatically | Team classification accuracy > 95% |
| P0 | Compute possession continuously | Possession accuracy within ±3% of manual ground truth |
| P0 | Detect goals automatically | Goal recall > 99% (zero misses acceptable) |
| P1 | Compute 20+ match statistics | All stats within ±5% of manual annotation |
| P1 | Stream live stats in under 200ms | P99 stat delivery latency < 200ms |
| P1 | Generate player-level deep stats | Distance, speed, heatmap, actions per player |
| P2 | Detect formations automatically | Formation accuracy > 85% on 10 most common formations |
| P2 | Calculate expected goals (xG) | AUC > 0.78 on held-out shot dataset |
| P2 | Generate AI match commentary | Commentary relevance score > 4.2/5 in user testing |
| P3 | Support live camera input | Sub-second latency on WebSocket delivery |
| P3 | Extract highlight reels automatically | Precision > 80% on user-rated highlights |

### 2.4 Non-Goals (v1.0)

- **Not a player tracking app** requiring GPS wearables
- **No referee-grade offside VAR** (indicative only)
- **No audio/crowd analysis** in v1.0 (planned v2.0)
- **No mobile app** (API-only; frontend is a separate product)
- **No live broadcast encoding** (clients receive stats, not encoded video streams)

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                               │
│   Mobile App    Coaching Dashboard    Broadcast Overlay    3rd Party │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTPS / WSS
┌────────────────────────────▼────────────────────────────────────────┐
│                      API GATEWAY (FastAPI)                          │
│   Auth Middleware · Rate Limiting · Request Validation · OpenAPI    │
└──┬──────────────┬────────────────┬──────────────────────────┬───────┘
   │              │                │                          │
   ▼              ▼                ▼                          ▼
[Auth Svc]  [Ingestion Svc]  [WebSocket Hub]         [Report Svc]
   │              │                │                          │
   │         ┌────▼────┐           │                          │
   │         │  MinIO  │           │                          │
   │         │(Storage)│           │                          │
   │         └────┬────┘           │                          │
   │              │                │                          │
   │         ┌────▼────────────────▼──────────────────────────▼─────┐
   │         │                  Redis                               │
   │         │   Streams (job queue) · Pub/Sub · Cache · Sessions   │
   │         └────┬─────────────────────────────────────────────────┘
   │              │
   │    ┌─────────▼──────────┐
   │    │  Inference Service  │   ← GPU Worker (YOLOv8 + ByteTrack)
   │    │  (CV Pipeline)      │
   │    └─────────┬──────────┘
   │              │  detections stream
   │    ┌─────────▼──────────┐
   │    │   Stats Engine     │   ← Pure Python / NumPy
   │    └─────────┬──────────┘
   │              │  events + stats
   │    ┌─────────▼──────────┐
   │    │  PostgreSQL 16      │
   │    │  (Primary DB)       │
   │    └────────────────────┘
   │
   └──── python-jose JWT · bcrypt · API Keys
```

### 3.2 Service Inventory

| Service | Language | Main Libraries | Role |
|---|---|---|---|
| API Gateway | Python 3.11 | FastAPI, Pydantic v2, uvicorn | Single public entry point |
| Auth Service | Python 3.11 | python-jose, passlib, bcrypt | JWT + API key management |
| Ingestion Service | Python 3.11 | aiofiles, ffmpeg-python, boto3 | Video upload, validation, job dispatch |
| Inference Engine | Python 3.11 | ultralytics, supervision, OpenCV | Frame-level CV processing |
| Stats Engine | Python 3.11 | NumPy, SciPy, scikit-learn | Stat accumulation and event detection |
| WebSocket Hub | Python 3.11 | FastAPI WS, Redis pub/sub | Real-time stat delivery |
| Player Analytics | Python 3.11 | NumPy, pandas | Deep per-player aggregation |
| xG Module | Python 3.11 | XGBoost, scikit-learn | Expected goals prediction |
| Commentary AI | Python 3.11 | openai, llama-cpp-python | LLM-based match commentary |
| Tactical Analysis | Python 3.11 | NumPy, scikit-learn, SciPy | Formation, pressing, heatmaps |
| Report Service | Python 3.11 | WeasyPrint, Jinja2, matplotlib | PDF report generation |
| Chat Assistant | Python 3.11 | openai, MCP tools | Conversational match Q&A |
| Highlight Extractor | Python 3.11 | OpenCV, FFmpeg | Highlight clip compilation |
| Camera Fusion | Python 3.11 | OpenCV, NumPy | Multi-camera sync and merge |

### 3.3 Data Flow — Video Upload (Async)

```
1. Client → POST /api/v1/matches/upload  (multipart video file)
2. API Gateway validates JWT + file type + file size
3. Ingestion Service streams file to MinIO (chunked upload)
4. Ingestion Service creates match record in PostgreSQL (status=queued)
5. Ingestion Service publishes job to Redis Stream: queue:inference
6. API responds immediately: { match_id, status: "queued" }

7. Inference Worker picks up job from Redis Stream
8. Inference Worker reads video from MinIO frame by frame (OpenCV)
9. For each frame:
   a. YOLOv8 → raw detections
   b. ByteTrack → tracked detections with persistent IDs
   c. KMeans → team label per detection
   d. Homography → pixel→metre coordinates
   e. Kalman Filter → smoothed ball position
   f. Pose estimation → keypoints per player
   g. SlowFast → action label per player clip
   h. Write frame record to PostgreSQL (frames table)
   i. Publish frame detections to Redis: queue:stats

10. Stats Engine picks up from Redis Stream: queue:stats
11. Stats Engine updates all accumulators (possession, distances, events, etc.)
12. Stats Engine writes stats snapshot to PostgreSQL + Redis cache
13. Stats Engine publishes update to: pubsub:match:{match_id}

14. WebSocket Hub receives from pub/sub, delivers to connected clients
15. REST clients poll GET /api/v1/matches/{match_id}/stats

16. On last frame: Inference marks match status=completed
17. Celery task triggered: generate annotated video + PDF report
18. Report uploaded to MinIO; download URL written to matches table
```

### 3.4 Data Flow — Live Camera Stream

```
1. Client → POST /api/v1/matches/live/start  → returns { session_id }
2. Client opens WebSocket: /ws/live/{session_id}/ingest
3. Client streams raw JPEG frames (base64) or MJPEG over WebSocket
4. API Gateway forwards frames to Inference Worker via Redis
5. Inference Worker processes each frame (steps 9a–9h above)
6. Stats pushed back to client on WebSocket /ws/live/{session_id}/stats
   within 200ms of frame receipt
7. Client → POST /api/v1/matches/live/{session_id}/stop
8. Session finalised: composite video assembled, report generated
```

---

## 4. Service 1 — Authentication & User Management

### 4.1 Overview

Handles user registration, login, JWT issuance/refresh/revocation, API key lifecycle, and user profile management. All other services delegate authentication decisions to this service. Tokens use **RS256** (asymmetric signing) so downstream services can verify without a secret.

### 4.2 API Endpoints

| Method | Endpoint | Auth | Request Body | Response | Description |
|---|---|---|---|---|---|
| `POST` | `/api/v1/auth/register` | None | `RegisterRequest` | `UserResponse` | Create a new user account |
| `POST` | `/api/v1/auth/login` | None | `LoginRequest` | `TokenResponse` | Authenticate; receive access + refresh tokens |
| `POST` | `/api/v1/auth/refresh` | Refresh JWT | — | `TokenResponse` | Exchange refresh token for new access token |
| `POST` | `/api/v1/auth/logout` | JWT | — | `204 No Content` | Invalidate current refresh token in Redis blacklist |
| `GET` | `/api/v1/auth/me` | JWT | — | `UserResponse` | Return authenticated user's profile |
| `PATCH` | `/api/v1/auth/me` | JWT | `UpdateUserRequest` | `UserResponse` | Update name, password, preferences |
| `DELETE` | `/api/v1/auth/me` | JWT | — | `204 No Content` | Soft-delete account (sets is_active=false) |
| `POST` | `/api/v1/auth/api-keys` | JWT | `CreateApiKeyRequest` | `ApiKeyResponse` | Generate new API key; key shown once only |
| `GET` | `/api/v1/auth/api-keys` | JWT | — | `ApiKeyListResponse` | List all API keys (hashed; no plaintext) |
| `DELETE` | `/api/v1/auth/api-keys/{key_id}` | JWT | — | `204 No Content` | Revoke an API key immediately |
| `POST` | `/api/v1/auth/password-reset/request` | None | `{ email }` | `202 Accepted` | Send password reset email with signed token |
| `POST` | `/api/v1/auth/password-reset/confirm` | None | `PasswordResetRequest` | `204 No Content` | Verify reset token and set new password |

### 4.3 Request / Response Schemas

#### `RegisterRequest`
```json
{
  "email": "string (valid email, max 255)",
  "password": "string (min 8, max 128, must contain uppercase + digit)",
  "full_name": "string (optional, max 100)",
  "organisation": "string (optional, max 100)"
}
```

#### `LoginRequest`
```json
{
  "email": "string",
  "password": "string"
}
```

#### `TokenResponse`
```json
{
  "access_token": "string (JWT, expires in 30 minutes)",
  "refresh_token": "string (opaque, expires in 30 days)",
  "token_type": "Bearer",
  "expires_in": 1800
}
```

#### `UserResponse`
```json
{
  "id": "uuid",
  "email": "string",
  "full_name": "string | null",
  "organisation": "string | null",
  "plan": "free | pro | enterprise",
  "created_at": "ISO 8601 timestamp",
  "matches_count": "integer",
  "storage_used_bytes": "integer"
}
```

#### `ApiKeyResponse`
```json
{
  "id": "uuid",
  "label": "string",
  "key": "fiq_live_xxxxxxxxxxxxxxxxxxxxxxxx  (only shown on creation)",
  "prefix": "fiq_live_xxxx",
  "created_at": "ISO 8601",
  "last_used_at": "ISO 8601 | null",
  "expires_at": "ISO 8601 | null"
}
```

### 4.4 Data Schema — `users` Table

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `UUID` | PK, default gen_random_uuid() | Primary key |
| `email` | `VARCHAR(255)` | UNIQUE, NOT NULL | Login identifier |
| `password_hash` | `VARCHAR(60)` | NOT NULL | bcrypt hash (cost=12) |
| `full_name` | `VARCHAR(100)` | NULLABLE | Display name |
| `organisation` | `VARCHAR(100)` | NULLABLE | Club or company name |
| `plan` | `ENUM` | NOT NULL, default 'free' | free / pro / enterprise |
| `is_active` | `BOOLEAN` | NOT NULL, default true | Soft delete flag |
| `email_verified` | `BOOLEAN` | NOT NULL, default false | Email confirmation status |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, default now() | Registration timestamp |
| `updated_at` | `TIMESTAMPTZ` | NOT NULL, default now() | Last profile update |
| `last_login_at` | `TIMESTAMPTZ` | NULLABLE | Last successful login |
| `storage_quota_bytes` | `BIGINT` | NOT NULL, default 10GB | Storage limit per plan |

### 4.5 Data Schema — `api_keys` Table

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `UUID` | PK | Key identifier |
| `user_id` | `UUID` | FK → users.id, ON DELETE CASCADE | Owner |
| `label` | `VARCHAR(100)` | NOT NULL | Human-readable key name |
| `key_hash` | `VARCHAR(64)` | NOT NULL, UNIQUE | SHA-256 hash of key |
| `key_prefix` | `VARCHAR(20)` | NOT NULL | First chars for display |
| `is_active` | `BOOLEAN` | NOT NULL, default true | Revocation flag |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | Creation time |
| `last_used_at` | `TIMESTAMPTZ` | NULLABLE | Last usage |
| `expires_at` | `TIMESTAMPTZ` | NULLABLE | Optional expiry |

### 4.6 JWT Claims Structure

```json
{
  "sub": "user-uuid",
  "email": "user@example.com",
  "plan": "pro",
  "iat": 1714000000,
  "exp": 1714001800,
  "jti": "unique-token-id",
  "type": "access"
}
```

---

## 5. Service 2 — Match Ingestion

### 5.1 Overview

Accepts video file uploads and manages the lifecycle of each match from upload through processing to completion. Validates video formats, extracts metadata using FFprobe, stores the file in MinIO, creates the match record, and dispatches the CV job.

### 5.2 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/matches/upload` | JWT | Upload a recorded match video (multipart/form-data) |
| `POST` | `/api/v1/matches/upload/url` | JWT | Ingest a match from a remote URL (async download) |
| `POST` | `/api/v1/matches/live/start` | JWT | Start a live analysis session |
| `POST` | `/api/v1/matches/live/{session_id}/stop` | JWT | End live session and trigger report |
| `GET` | `/api/v1/matches` | JWT | List user's matches (paginated, filterable) |
| `GET` | `/api/v1/matches/{match_id}` | JWT | Get match metadata + status |
| `PATCH` | `/api/v1/matches/{match_id}` | JWT | Update match title, team names |
| `DELETE` | `/api/v1/matches/{match_id}` | JWT | Delete match + all associated data + MinIO files |
| `GET` | `/api/v1/matches/{match_id}/status` | JWT | Poll processing progress (0.0–1.0) |
| `GET` | `/api/v1/matches/{match_id}/frames/{frame_no}` | JWT | Fetch annotated JPEG for a specific frame |
| `GET` | `/api/v1/matches/{match_id}/video/annotated` | JWT | Download annotated output video (MP4) |
| `GET` | `/api/v1/matches/{match_id}/thumbnail` | JWT | Get match thumbnail (frame at 30s) |
| `POST` | `/api/v1/matches/{match_id}/reprocess` | JWT | Re-run CV pipeline with new settings |

### 5.3 Upload Request — Multipart Fields

| Field | Type | Required | Validation |
|---|---|---|---|
| `file` | `binary` | Yes | MP4 / MKV / AVI / MOV; max 10 GB |
| `title` | `string` | Yes | Max 200 chars |
| `team_a_name` | `string` | No | Max 100 chars; defaults to "Team A" |
| `team_b_name` | `string` | No | Max 100 chars; defaults to "Team B" |
| `match_date` | `date` | No | ISO 8601 date |
| `venue` | `string` | No | Max 200 chars |
| `competition` | `string` | No | Max 200 chars |
| `analysis_config` | `JSON string` | No | See AnalysisConfig schema below |

### 5.4 AnalysisConfig Schema

```json
{
  "enable_pose_estimation": true,
  "enable_action_recognition": true,
  "enable_xg_calculation": true,
  "enable_commentary": false,
  "team_a_jersey_hint": "#FF0000",
  "team_b_jersey_hint": "#0000FF",
  "field_calibration_mode": "auto | manual | skip",
  "output_annotated_video": true,
  "detection_confidence_threshold": 0.45,
  "tracking_max_age": 30,
  "frame_skip": 1,
  "homography_corners": [
    {"x": 0, "y": 0}, {"x": 1920, "y": 0},
    {"x": 1920, "y": 1080}, {"x": 0, "y": 1080}
  ]
}
```

### 5.5 Match Schema — Full

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `UUID` | PK | Match identifier |
| `user_id` | `UUID` | FK → users.id | Owner |
| `title` | `VARCHAR(200)` | NOT NULL | Human-readable match title |
| `team_a_name` | `VARCHAR(100)` | default 'Team A' | Team A label |
| `team_b_name` | `VARCHAR(100)` | default 'Team B' | Team B label |
| `team_a_jersey_color` | `VARCHAR(7)` | NULLABLE | Detected dominant jersey hex |
| `team_b_jersey_color` | `VARCHAR(7)` | NULLABLE | Detected dominant jersey hex |
| `match_date` | `DATE` | NULLABLE | When the match was played |
| `venue` | `VARCHAR(200)` | NULLABLE | Stadium / pitch name |
| `competition` | `VARCHAR(200)` | NULLABLE | League / cup name |
| `video_url` | `TEXT` | NOT NULL | MinIO URL of original file |
| `annotated_url` | `TEXT` | NULLABLE | MinIO URL of annotated output |
| `thumbnail_url` | `TEXT` | NULLABLE | MinIO URL of thumbnail JPEG |
| `file_size_bytes` | `BIGINT` | NOT NULL | Original file size |
| `duration_seconds` | `FLOAT` | NULLABLE | Total video duration |
| `fps` | `FLOAT` | NOT NULL | Source video FPS |
| `resolution_w` | `INTEGER` | NOT NULL | Frame width in pixels |
| `resolution_h` | `INTEGER` | NOT NULL | Frame height in pixels |
| `total_frames` | `INTEGER` | NULLABLE | Total frame count |
| `processed_frames` | `INTEGER` | default 0 | Frames processed so far |
| `status` | `ENUM` | NOT NULL | queued / processing / completed / failed / cancelled |
| `failure_reason` | `TEXT` | NULLABLE | Error message if status=failed |
| `is_live` | `BOOLEAN` | default false | True if sourced from live stream |
| `analysis_config` | `JSONB` | NULLABLE | AnalysisConfig used for this run |
| `homography_matrix` | `FLOAT[]` | NULLABLE | 9-element flat 3×3 matrix |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | Upload timestamp |
| `processing_started_at` | `TIMESTAMPTZ` | NULLABLE | When Inference Worker began |
| `completed_at` | `TIMESTAMPTZ` | NULLABLE | When processing finished |

### 5.6 List Matches — Query Parameters

```
GET /api/v1/matches?page=1&limit=20&status=completed&sort=created_at&order=desc&search=final
```

| Param | Type | Default | Description |
|---|---|---|---|
| `page` | integer | 1 | Page number |
| `limit` | integer | 20 | Items per page (max 100) |
| `status` | enum | all | Filter by processing status |
| `sort` | string | created_at | Sort field |
| `order` | enum | desc | asc / desc |
| `search` | string | — | Full-text search in title, venue, competition |
| `from_date` | date | — | Filter matches on or after this date |
| `to_date` | date | — | Filter matches on or before this date |
| `is_live` | boolean | — | Filter live vs uploaded matches |

---

## 6. Service 3 — Inference Engine (CV Pipeline)

### 6.1 Overview

The core computer vision pipeline. This is the GPU-intensive service that reads video frames and extracts all structured data about what is happening on the pitch. It is designed as a **worker process** consuming jobs from Redis Streams, not a public HTTP service.

### 6.2 CV Pipeline — Stage-by-Stage

#### Stage 1: Frame Reading & Pre-processing

```python
# OpenCV frame reader with configurable skip
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    
    # Resize to 1280×720 for inference if source is larger
    # Preserve aspect ratio; pad with letterbox
    frame_preprocessed = letterbox(frame, new_shape=(1280, 720))
```

#### Stage 2: Object Detection — YOLOv8x

**Model:** `yolov8x.pt` fine-tuned on SoccerNet + custom labelled dataset  
**Classes detected:**
- `0` — player
- `1` — ball
- `2` — referee
- `3` — goalkeeper (optional separate class)
- `4` — goalpost / goal frame

**Configuration:**
```python
model = YOLO("yolov8x_football.pt")
results = model.predict(
    source=frame,
    conf=0.45,          # confidence threshold
    iou=0.45,           # NMS IoU threshold
    max_det=50,         # max 22 players + ball + refs + margin
    device="cuda:0",
    half=True,          # FP16 for speed on GPU
    verbose=False,
)
```

**Output per detection:**
```python
{
    "bbox_xyxy": [x1, y1, x2, y2],   # pixel coords
    "confidence": 0.87,
    "class_id": 0,                    # 0=player
    "class_name": "player"
}
```

#### Stage 3: Multi-Object Tracking — ByteTrack

**Library:** `supervision.ByteTrack`  
**Purpose:** Assigns persistent `track_id` across frames; handles occlusion and re-entry.

```python
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,    # frames to keep lost track alive
    minimum_matching_threshold=0.8,
    frame_rate=fps,
)
detections = sv.Detections.from_ultralytics(results[0])
detections = tracker.update_with_detections(detections)
# detections.tracker_id now populated with persistent IDs
```

#### Stage 4: Team Classification — KMeans on Jersey Crops

**Goal:** Assign each player detection to `team_a`, `team_b`, or `referee`  
**Method:**

```python
# Step 1: For each player detection, crop the jersey region (top 40% of bbox)
def crop_jersey(frame, bbox):
    x1, y1, x2, y2 = bbox
    jersey_h = int((y2 - y1) * 0.4)
    return frame[y1:y1+jersey_h, x1:x2]

# Step 2: Convert crop to HSV, flatten pixels, filter out green (grass)
def extract_jersey_colors(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # mask out green (grass contamination)
    non_grass = mask_green(hsv)
    return hsv[non_grass].reshape(-1, 3)

# Step 3: KMeans with k=3 across ALL player crops in a frame batch
# Clusters: team_a, team_b, referee
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(all_player_colors)

# Step 4: Assign labels; referee cluster identified by having
# a different colour profile (typically black/yellow)
```

**Temporal smoothing:** Team label votes across last 10 frames per `track_id` before finalising label. This prevents flickering.

#### Stage 5: Field Homography — Perspective Transform

**Goal:** Map pixel coordinates → real-world metres on a standard 105m × 68m pitch.

**Two modes:**
1. **Manual** — client provides 4 pixel coordinates of field corners; API computes homography matrix once.
2. **Auto** — field line detector (Canny + Hough lines) locates field boundary, centre circle, and penalty area corners to fit homography automatically.

```python
# Manual mode
src_points = np.array(config.homography_corners, dtype=np.float32)
dst_points = np.array([
    [0, 0], [105, 0], [105, 68], [0, 68]  # real-world metres
], dtype=np.float32)
H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# Apply to any pixel point
def pixel_to_metres(px, py, H):
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return result[0][0]  # (x_m, y_m)
```

#### Stage 6: Ball Trajectory — Kalman Filter + Cubic Spline

**Problem:** Ball is frequently occluded (behind players, off-screen). Raw detections are noisy.

```python
# Kalman filter state: [x, y, vx, vy] (position + velocity)
from filterpy.kalman import KalmanFilter

kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])  # state transition
kf.H = np.array([[1,0,0,0],[0,1,0,0]])                       # measurement
kf.R = np.diag([10, 10])       # measurement noise
kf.Q = np.eye(4) * 0.1         # process noise

# When ball detected: kf.update(ball_px)
# When ball not detected: kf.predict() only → estimated position used
# Fill gaps with cubic spline interpolation over detected positions
```

#### Stage 7: Pose Estimation — YOLOv8-pose

**Model:** `yolov8x-pose.pt`  
**Output:** 17 COCO keypoints per player per frame

```
0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear
5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow
9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip
13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
```

Each keypoint: `[x_px, y_px, confidence]`

**Use cases:** Action recognition feature input, body orientation, tackle/foul detection

#### Stage 8: Action Recognition — SlowFast Network

**Model:** SlowFast-R50 fine-tuned on AVA + SoccerNet Actions dataset  
**Input:** 16-frame temporal clip (±8 frames) centred on current frame, per player crop  
**Output:** Action class probability distribution

**Action classes:**
```
0: standing / jogging
1: sprinting
2: passing (kick)
3: receiving (controlling ball)
4: shooting
5: heading
6: tackling (sliding / standing)
7: dribbling
8: goalkeeper_save
9: celebrating
```

**Inference schedule:** Action recognition runs every 8 frames (not every frame) to maintain throughput. Results interpolated between inference frames.

### 6.3 Frame Detection Schema — Database

#### `frames` Table

| Column | Type | Description |
|---|---|---|
| `id` | `UUID` | Frame record identifier |
| `match_id` | `UUID` | Parent match (FK) |
| `frame_number` | `INTEGER` | Sequential 0-based index |
| `timestamp_ms` | `FLOAT` | Position in video (ms) |
| `detections` | `JSONB` | Array of DetectionObject |
| `ball_position_px` | `POINT` | Ball pixel centroid (nullable) |
| `ball_position_m` | `POINT` | Ball real-world position in metres (nullable) |
| `ball_velocity_mps` | `FLOAT` | Ball speed in metres/second |
| `ball_confidence` | `FLOAT` | Detection confidence for ball |
| `homography_matrix` | `FLOAT[]` | 9-element flat 3×3 (nullable) |
| `processing_time_ms` | `FLOAT` | Time to process this frame |

#### `DetectionObject` (JSONB element)

```json
{
  "track_id": 7,
  "class_label": "player",
  "team": "team_a",
  "bbox_norm": [0.42, 0.35, 0.48, 0.62],
  "bbox_px": [806, 378, 922, 669],
  "confidence": 0.91,
  "position_px": {"x": 864, "y": 524},
  "position_m": {"x": 61.2, "y": 28.7},
  "speed_kmh": 18.4,
  "acceleration_ms2": 1.2,
  "keypoints": [
    [864, 385, 0.92], [858, 382, 0.89],
    ...
  ],
  "action": "dribbling",
  "action_confidence": 0.78,
  "jersey_color_hsv": [12, 220, 180]
}
```

### 6.4 Inference Service Internal Configuration

```python
class InferenceConfig:
    detection_model: str = "yolov8x_football.pt"
    pose_model: str = "yolov8x-pose.pt"
    action_model: str = "slowfast_r50_football.pt"
    detection_confidence: float = 0.45
    tracking_max_age: int = 30
    tracking_min_hits: int = 3
    team_kmeans_clusters: int = 3
    pose_enabled: bool = True
    action_enabled: bool = True
    frame_skip: int = 1           # process every Nth frame
    action_clip_frames: int = 16
    action_inference_every: int = 8
    batch_size: int = 1           # frames per GPU batch
    device: str = "cuda:0"
    fp16: bool = True
```

---

## 7. Service 4 — Statistics Engine

### 7.1 Overview

A pure-Python/NumPy service that consumes frame detections from Redis Streams and maintains running accumulators for every match statistic. Designed to be **stateless** between runs — all state is persisted to PostgreSQL + Redis, so the engine can restart mid-match without data loss.

### 7.2 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/matches/{match_id}/stats` | JWT | Full stats snapshot (latest) |
| `GET` | `/api/v1/matches/{match_id}/stats/live` | JWT | Ultra-low-latency snapshot (from Redis cache) |
| `GET` | `/api/v1/matches/{match_id}/stats/possession` | JWT | Possession time-series (1-second buckets) |
| `GET` | `/api/v1/matches/{match_id}/stats/possession/zones` | JWT | Possession broken down by pitch thirds |
| `GET` | `/api/v1/matches/{match_id}/stats/heatmap/{team}` | JWT | 2D heatmap array for team (32×52 grid) |
| `GET` | `/api/v1/matches/{match_id}/stats/heatmap/player/{track_id}` | JWT | Heatmap for individual player |
| `GET` | `/api/v1/matches/{match_id}/stats/events` | JWT | Timeline of all detected events |
| `GET` | `/api/v1/matches/{match_id}/stats/events?type=shot` | JWT | Filter events by type |
| `GET` | `/api/v1/matches/{match_id}/stats/players` | JWT | All player stats summary |
| `GET` | `/api/v1/matches/{match_id}/stats/players/{track_id}` | JWT | One player's full stats |
| `GET` | `/api/v1/matches/{match_id}/stats/formations` | JWT | Formation snapshots over time |
| `GET` | `/api/v1/matches/{match_id}/stats/momentum` | JWT | Momentum index time-series |
| `GET` | `/api/v1/matches/{match_id}/stats/passes/network` | JWT | Pass network graph (nodes + edges) |
| `GET` | `/api/v1/matches/{match_id}/stats/passes/map` | JWT | Individual pass start/end coordinates |
| `GET` | `/api/v1/matches/{match_id}/stats/shots/map` | JWT | Shot locations + xG + outcome |
| `GET` | `/api/v1/matches/{match_id}/stats/pressing` | JWT | PPDA pressing intensity over time |
| `GET` | `/api/v1/matches/{match_id}/stats/defensive_line` | JWT | Defensive line height time-series |
| `GET` | `/api/v1/matches/{match_id}/stats/xg/timeline` | JWT | Cumulative xG over match time |
| `GET` | `/api/v1/matches/{match_id}/stats/speed/top10` | JWT | Top 10 fastest speed moments in the match |

### 7.3 Match Statistics Schema — `match_stats` Table

| Column | Type | Description |
|---|---|---|
| `id` | `UUID` | Stats record |
| `match_id` | `UUID` | Parent match (FK) |
| `updated_at` | `TIMESTAMPTZ` | Last update time |
| `last_frame_number` | `INTEGER` | Most recent frame included |
| **Possession** | | |
| `possession_a_pct` | `FLOAT` | Team A possession % |
| `possession_b_pct` | `FLOAT` | Team B possession % |
| `possession_a_seconds` | `FLOAT` | Seconds Team A held the ball |
| `possession_b_seconds` | `FLOAT` | Seconds Team B held the ball |
| `possession_a_own_half` | `FLOAT` | % of possession in own half |
| `possession_a_opp_half` | `FLOAT` | % of possession in opponent half |
| `possession_b_own_half` | `FLOAT` | Same for Team B |
| `possession_b_opp_half` | `FLOAT` | Same for Team B |
| **Scoring** | | |
| `goals_a` | `INTEGER` | Goals by Team A |
| `goals_b` | `INTEGER` | Goals by Team B |
| `shots_a` | `INTEGER` | Total shots by Team A |
| `shots_on_target_a` | `INTEGER` | Shots on target by Team A |
| `shots_off_target_a` | `INTEGER` | Shots off target by Team A |
| `shots_blocked_a` | `INTEGER` | Shots blocked by Team A |
| `shots_b` | `INTEGER` | Total shots by Team B |
| `shots_on_target_b` | `INTEGER` | Shots on target by Team B |
| `shots_off_target_b` | `INTEGER` | Shots off target by Team B |
| `shots_blocked_b` | `INTEGER` | Shots blocked by Team B |
| `xg_a` | `FLOAT` | Expected goals Team A |
| `xg_b` | `FLOAT` | Expected goals Team B |
| **Passing** | | |
| `passes_attempted_a` | `INTEGER` | Total pass attempts Team A |
| `passes_completed_a` | `INTEGER` | Successful passes Team A |
| `pass_accuracy_a` | `FLOAT` | Completion % Team A |
| `passes_attempted_b` | `INTEGER` | Total pass attempts Team B |
| `passes_completed_b` | `INTEGER` | Successful passes Team B |
| `pass_accuracy_b` | `FLOAT` | Completion % Team B |
| `long_balls_a` | `INTEGER` | Passes > 32m by Team A |
| `long_balls_b` | `INTEGER` | Passes > 32m by Team B |
| `crosses_a` | `INTEGER` | Crosses from wide areas by Team A |
| `crosses_b` | `INTEGER` | Crosses from wide areas by Team B |
| **Defensive** | | |
| `tackles_a` | `INTEGER` | Tackles by Team A |
| `tackles_won_a` | `INTEGER` | Tackles won by Team A |
| `tackles_b` | `INTEGER` | Tackles by Team B |
| `tackles_won_b` | `INTEGER` | Tackles won by Team B |
| `fouls_a` | `INTEGER` | Fouls committed by Team A |
| `fouls_b` | `INTEGER` | Fouls committed by Team B |
| `interceptions_a` | `INTEGER` | Ball interceptions by Team A |
| `interceptions_b` | `INTEGER` | Ball interceptions by Team B |
| `clearances_a` | `INTEGER` | Goal-line/area clearances by Team A |
| `clearances_b` | `INTEGER` | Goal-line/area clearances by Team B |
| **Set Pieces** | | |
| `corners_a` | `INTEGER` | Corners taken by Team A |
| `corners_b` | `INTEGER` | Corners taken by Team B |
| `offsides_a` | `INTEGER` | Offside calls against Team A |
| `offsides_b` | `INTEGER` | Offside calls against Team B |
| `free_kicks_a` | `INTEGER` | Free kicks awarded to Team A |
| `free_kicks_b` | `INTEGER` | Free kicks awarded to Team B |
| **Physical** | | |
| `total_distance_a_km` | `FLOAT` | Total km covered by Team A |
| `total_distance_b_km` | `FLOAT` | Total km covered by Team B |
| `avg_speed_a_kmh` | `FLOAT` | Average speed Team A |
| `avg_speed_b_kmh` | `FLOAT` | Average speed Team B |
| `max_speed_a_kmh` | `FLOAT` | Peak speed any Team A player |
| `max_speed_b_kmh` | `FLOAT` | Peak speed any Team B player |
| `sprints_a` | `INTEGER` | Sprint count Team A (bursts > 25 km/h) |
| `sprints_b` | `INTEGER` | Sprint count Team B |
| `high_intensity_runs_a` | `INTEGER` | Runs > 19.8 km/h by Team A |
| `high_intensity_runs_b` | `INTEGER` | Runs > 19.8 km/h by Team B |
| **Tactical** | | |
| `formation_a` | `VARCHAR(10)` | Latest formation e.g. "4-3-3" |
| `formation_b` | `VARCHAR(10)` | Latest formation |
| `pressing_intensity_a` | `FLOAT` | PPDA score Team A (lower = more pressing) |
| `pressing_intensity_b` | `FLOAT` | PPDA score Team B |
| `defensive_line_height_a` | `FLOAT` | Average defensive line y-coord in metres |
| `defensive_line_height_b` | `FLOAT` | Average defensive line y-coord in metres |
| `momentum_a` | `FLOAT` | Current momentum index 0–100 |
| `momentum_b` | `FLOAT` | Current momentum index 0–100 |

### 7.4 Event Schema — `events` Table

| Column | Type | Description |
|---|---|---|
| `id` | `UUID` | Event identifier |
| `match_id` | `UUID` | Parent match |
| `type` | `ENUM` | goal / shot / pass / tackle / foul / corner / offside / save / dribble / interception / clearance / free_kick / cross / header |
| `timestamp_ms` | `FLOAT` | Video position |
| `match_minute` | `INTEGER` | Computed match minute |
| `team` | `ENUM` | team_a / team_b |
| `player_track_id` | `INTEGER` | Acting player (nullable) |
| `secondary_player_track_id` | `INTEGER` | Receiver / tackled player (nullable) |
| `location_px` | `POINT` | Pixel coordinates |
| `location_m` | `POINT` | Real-world field coordinates in metres |
| `outcome` | `ENUM` | success / failure / on_target / off_target / blocked / saved / wide / over |
| `xg_value` | `FLOAT` | xG for shot events |
| `pass_distance_m` | `FLOAT` | Distance for pass events |
| `shot_distance_m` | `FLOAT` | Distance for shot events |
| `shot_angle_deg` | `FLOAT` | Angle for shot events |
| `body_part` | `ENUM` | right_foot / left_foot / head (for shots/headers) |
| `confidence` | `FLOAT` | Model detection confidence |
| `frame_number` | `INTEGER` | Source frame index |

### 7.5 Stat Calculation Methods

#### Possession Calculation

```python
def update_possession(frame_detections, ball_position_m):
    """
    Ball ownership: find the player with smallest Euclidean distance
    to the ball. If closest player is within 2 metres → that team owns.
    If ball > 2m from all players → contested (neither team credited).
    """
    min_dist = float('inf')
    owner_team = None
    
    for det in frame_detections:
        if det.class_label != 'player':
            continue
        dist = euclidean(det.position_m, ball_position_m)
        if dist < min_dist:
            min_dist = dist
            owner_team = det.team
    
    if min_dist <= 2.0:  # 2-metre threshold
        possession_frames[owner_team] += 1
    
    total = sum(possession_frames.values())
    return {
        'team_a': possession_frames['team_a'] / total * 100,
        'team_b': possession_frames['team_b'] / total * 100,
    }
```

#### Speed Calculation

```python
def calc_speed(pos_curr_m, pos_prev_m, fps):
    dist_m = euclidean(pos_curr_m, pos_prev_m)
    speed_mps = dist_m * fps           # metres per second
    return speed_mps * 3.6             # convert to km/h

# Smoothed with rolling 5-frame average to reduce GPS-like noise
```

#### Pass Detection

```python
def detect_pass(prev_owner_id, curr_owner_id, team_lookup):
    """
    A pass is detected when:
    1. Ball transitions from player A to player B
    2. Both players belong to the SAME team
    3. The transition happens within 2 seconds (not a tackle)
    """
    if prev_owner_id == curr_owner_id:
        return None
    if team_lookup[prev_owner_id] != team_lookup[curr_owner_id]:
        return None   # possession change = tackle/interception, not pass
    return PassEvent(from_id=prev_owner_id, to_id=curr_owner_id)
```

---

## 8. Service 5 — Live Streaming & WebSocket

### 8.1 Overview

Manages WebSocket connections for real-time stat delivery. Uses **Redis Pub/Sub** as the intermediary between the Stats Engine (which writes) and connected WebSocket clients (which read). Supports multiple clients per match simultaneously.

### 8.2 WebSocket Endpoints

| Endpoint | Auth | Direction | Description |
|---|---|---|---|
| `/ws/live/{session_id}/ingest` | JWT (query param) | Client → Server | Live camera feed: client sends JPEG frames (base64) |
| `/ws/live/{session_id}/stats` | JWT | Server → Client | Live stats push for a live session |
| `/ws/match/{match_id}/stats` | JWT | Server → Client | Stats push during upload processing |
| `/ws/match/{match_id}/replay` | JWT | Server → Client | Replay processed match at original speed |
| `/ws/match/{match_id}/commentary` | JWT | Server → Client | Token-streamed AI commentary |

### 8.3 WebSocket Message Format — Server → Client

#### `stats_update` Message

```json
{
  "event": "stats_update",
  "match_id": "uuid",
  "frame_number": 1847,
  "timestamp_ms": 61566.7,
  "match_minute": 61,
  "score": { "team_a": 1, "team_b": 0 },
  "possession": {
    "team_a": 58.3,
    "team_b": 41.7,
    "current_holder": "team_a"
  },
  "ball": {
    "position_m": { "x": 45.2, "y": 28.1 },
    "speed_kmh": 62.4,
    "confidence": 0.88
  },
  "players": [
    {
      "track_id": 7,
      "team": "team_a",
      "position_m": { "x": 38.1, "y": 22.0 },
      "speed_kmh": 24.2,
      "action": "dribbling",
      "is_in_possession": true
    }
  ],
  "last_event": {
    "type": "shot",
    "team": "team_a",
    "player_track_id": 7,
    "outcome": "on_target",
    "xg": 0.34,
    "timestamp_ms": 61540.0
  },
  "stats_snapshot": {
    "shots_a": 7,
    "shots_on_target_a": 4,
    "shots_b": 3,
    "shots_on_target_b": 1,
    "passes_completed_a": 184,
    "passes_completed_b": 122,
    "xg_a": 1.82,
    "xg_b": 0.67
  }
}
```

#### `goal_detected` Message

```json
{
  "event": "goal_detected",
  "team": "team_a",
  "player_track_id": 11,
  "timestamp_ms": 61620.0,
  "match_minute": 62,
  "new_score": { "team_a": 2, "team_b": 0 },
  "xg": 0.71,
  "shot_location_m": { "x": 91.2, "y": 34.5 },
  "distance_m": 13.8,
  "body_part": "right_foot",
  "assist_by_track_id": 7
}
```

#### `processing_progress` Message

```json
{
  "event": "processing_progress",
  "match_id": "uuid",
  "progress": 0.47,
  "frames_processed": 4700,
  "total_frames": 10000,
  "eta_seconds": 68
}
```

### 8.4 Client → Server Frame Ingestion Format

```json
{
  "event": "frame",
  "session_id": "uuid",
  "frame_number": 1234,
  "timestamp_ms": 41200,
  "image_data": "base64_encoded_jpeg..."
}
```

---

## 9. Service 6 — Player Analytics

### 9.1 Overview

Aggregates frame-level detections into comprehensive per-player statistics. Runs as a batch aggregation after match completion but can also provide rolling estimates during processing.

### 9.2 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/matches/{match_id}/players` | JWT | All players summary list |
| `GET` | `/api/v1/matches/{match_id}/players/{track_id}` | JWT | Full stats for one player |
| `GET` | `/api/v1/matches/{match_id}/players/{track_id}/heatmap` | JWT | Spatial heatmap grid |
| `GET` | `/api/v1/matches/{match_id}/players/{track_id}/trajectory` | JWT | Movement path (x, y, t samples) |
| `GET` | `/api/v1/matches/{match_id}/players/{track_id}/actions` | JWT | Action event timeline |
| `GET` | `/api/v1/matches/{match_id}/players/{track_id}/speed/timeline` | JWT | Speed time-series |
| `GET` | `/api/v1/matches/{match_id}/players/{track_id}/passes` | JWT | All passes with start/end coords |
| `POST` | `/api/v1/players/compare` | JWT | Side-by-side comparison of two players |
| `GET` | `/api/v1/players/{track_id}/scout-report` | JWT | Download scouting PDF for one player |
| `GET` | `/api/v1/matches/{match_id}/players/leaderboard` | JWT | Top players ranked by any stat |

### 9.3 Player Stats Schema — `player_stats` Table

| Column | Type | Description |
|---|---|---|
| `id` | `UUID` | Record identifier |
| `match_id` | `UUID` | Parent match |
| `track_id` | `INTEGER` | ByteTrack persistent ID |
| `team` | `ENUM` | team_a / team_b / referee |
| `player_name` | `VARCHAR(100)` | Optional (from identity module) |
| `jersey_number` | `SMALLINT` | Detected by OCR (nullable) |
| `dominant_jersey_color` | `VARCHAR(7)` | Detected hex colour |
| `first_seen_frame` | `INTEGER` | First frame with this track_id |
| `last_seen_frame` | `INTEGER` | Last frame with this track_id |
| `frames_detected` | `INTEGER` | Total frames where this player appeared |
| `time_on_pitch_seconds` | `FLOAT` | Computed from first/last frame |
| **Physical** | | |
| `total_distance_km` | `FLOAT` | Total distance covered |
| `avg_speed_kmh` | `FLOAT` | Average speed when moving |
| `max_speed_kmh` | `FLOAT` | Peak speed |
| `max_speed_timestamp_ms` | `FLOAT` | When peak speed occurred |
| `sprints_count` | `INTEGER` | Sprint bursts (>25 km/h) |
| `high_intensity_runs` | `INTEGER` | Runs > 19.8 km/h |
| `walking_distance_km` | `FLOAT` | Distance at < 7 km/h |
| `jogging_distance_km` | `FLOAT` | Distance at 7–14 km/h |
| `running_distance_km` | `FLOAT` | Distance at 14–19.8 km/h |
| `sprinting_distance_km` | `FLOAT` | Distance at > 25 km/h |
| **Ball Skills** | | |
| `touches` | `INTEGER` | Total ball touches |
| `time_in_possession_s` | `FLOAT` | Seconds with ball within 1m |
| `passes_attempted` | `INTEGER` | Pass attempts |
| `passes_completed` | `INTEGER` | Successful passes |
| `pass_accuracy` | `FLOAT` | Completion % |
| `key_passes` | `INTEGER` | Passes leading to a shot |
| `long_passes_attempted` | `INTEGER` | Passes > 32m |
| `long_passes_completed` | `INTEGER` | Successful long passes |
| `crosses_attempted` | `INTEGER` | Wide crosses attempted |
| `crosses_completed` | `INTEGER` | Accurate crosses |
| `dribbles_attempted` | `INTEGER` | Dribble attempts |
| `dribbles_succeeded` | `INTEGER` | Successful dribbles |
| **Attacking** | | |
| `shots` | `INTEGER` | Total shots |
| `shots_on_target` | `INTEGER` | On-target shots |
| `goals` | `INTEGER` | Goals scored |
| `assists` | `INTEGER` | Goal assists |
| `xg_total` | `FLOAT` | Total xG from shots |
| `xg_per_shot` | `FLOAT` | Average xG per shot |
| **Defensive** | | |
| `tackles_attempted` | `INTEGER` | Total tackle attempts |
| `tackles_won` | `INTEGER` | Successful tackles |
| `interceptions` | `INTEGER` | Ball interceptions |
| `clearances` | `INTEGER` | Defensive clearances |
| `fouls_committed` | `INTEGER` | Fouls committed |
| `fouls_drawn` | `INTEGER` | Fouls drawn from opponent |
| `headers_won` | `INTEGER` | Aerial duels won |
| **Spatial** | | |
| `avg_position_m` | `POINT` | Average field position in metres |
| `heatmap` | `FLOAT[]` | 32×52 presence heatmap (1664 floats) |
| `territory_pct_own_half` | `FLOAT` | % time in own half |
| `territory_pct_opp_half` | `FLOAT` | % time in opponent half |
| `territory_pct_left_channel` | `FLOAT` | % time in left third of pitch width |
| `territory_pct_right_channel` | `FLOAT` | % time in right third |
| `territory_pct_central` | `FLOAT` | % time in central corridor |

### 9.4 Player Comparison Response

```json
{
  "player_a": {
    "track_id": 7,
    "team": "team_a",
    "stats": { ... }
  },
  "player_b": {
    "track_id": 14,
    "team": "team_b",
    "stats": { ... }
  },
  "comparison": {
    "distance_winner": "player_a",
    "max_speed_winner": "player_b",
    "pass_accuracy_winner": "player_a",
    "xg_winner": "player_a",
    "tackles_won_winner": "player_b"
  }
}
```

---

## 10. Service 7 — Expected Goals (xG) Module

### 10.1 Overview

Calculates the probability (0.0–1.0) that a given shot results in a goal, based on spatial, contextual, and situational features. The model is trained on 80,000+ historical shots with known outcomes.

### 10.2 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/xg/predict` | API Key | Real-time xG prediction for a shot |
| `GET` | `/api/v1/matches/{match_id}/xg/timeline` | JWT | Cumulative xG over time for both teams |
| `GET` | `/api/v1/matches/{match_id}/xg/shots` | JWT | Shot map: coords + xG + outcome per shot |
| `GET` | `/api/v1/matches/{match_id}/xg/summary` | JWT | Final xG totals with breakdown |

### 10.3 xG Feature Schema

| Feature | Type | Source | Description |
|---|---|---|---|
| `distance_m` | `float` | Homography | Distance from ball to goal centre |
| `angle_deg` | `float` | Computed | Angle to goal from shooter position (0=on goal line, 90=central) |
| `is_header` | `boolean` | Action model | True if header |
| `body_part` | `enum` | Action model | right_foot / left_foot / head |
| `assist_type` | `enum` | Event detector | open_play / cross / corner / free_kick / penalty / through_ball |
| `defenders_in_path` | `integer` | Detection | Defenders between shooter and goal at time of shot |
| `goalkeeper_off_line` | `boolean` | Detection | True if keeper > 2m from goal line |
| `goalkeeper_dist_m` | `float` | Detection | Distance from keeper to shooter |
| `is_first_touch` | `boolean` | Detection | True if shot on first touch after receiving |
| `pressure_rating` | `float` | Detection | Pressing intensity from nearest opponent (0–1) |
| `shot_speed_kmh` | `float` | Kalman | Ball speed at moment of shot |
| `game_minute` | `integer` | Match | Match minute (late game changes xG context) |
| `score_diff` | `integer` | Stats | Score difference at time of shot |

### 10.4 Model Details

**Model type:** XGBoost Classifier (primary) + Logistic Regression (baseline comparison)

```python
xg_model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    use_label_encoder=False,
)
# Training data: StatsBomb open dataset + Wyscout open data
# ~80,000 shots; ~9% conversion rate
# Target: AUC > 0.78, Brier score < 0.07
```

**xG Response:**
```json
{
  "xg": 0.34,
  "probability_goal": 0.34,
  "confidence_interval": [0.28, 0.41],
  "feature_importances": {
    "distance_m": 0.31,
    "angle_deg": 0.24,
    "defenders_in_path": 0.18,
    "body_part": 0.12,
    "goalkeeper_off_line": 0.09,
    "assist_type": 0.06
  }
}
```

---

## 11. Service 8 — AI Commentary Engine

### 11.1 Overview

Generates natural-language football commentary in real time from the structured event stream. Uses an LLM (GPT-4o via API, or Llama 3 70B hosted locally) with a carefully designed system prompt and rolling context window.

### 11.2 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/matches/{match_id}/commentary` | JWT | Full match commentary transcript |
| `GET` | `/api/v1/matches/{match_id}/commentary/highlights` | JWT | Top 10 commentary moments |
| `POST` | `/api/v1/commentary/generate` | API Key | Generate commentary for a single event on-demand |
| `PATCH` | `/api/v1/matches/{match_id}/commentary/style` | JWT | Change style (professional / excited / tactical / casual) |
| `GET` | `/api/v1/matches/{match_id}/commentary/audio` | JWT | Download TTS audio commentary (MP3) |
| `GET` | `/api/v1/matches/{match_id}/commentary/subtitles` | JWT | Download SRT subtitle file |

### 11.3 Commentary Schema — `commentary` Table

| Column | Type | Description |
|---|---|---|
| `id` | `UUID` | Commentary record |
| `match_id` | `UUID` | Parent match |
| `event_id` | `UUID` | Source event (FK, nullable) |
| `frame_number` | `INTEGER` | Source frame |
| `timestamp_ms` | `FLOAT` | Match time |
| `text` | `TEXT` | Generated commentary line |
| `style` | `ENUM` | professional / excited / tactical / casual |
| `language` | `VARCHAR(5)` | ISO 639-1 language code |
| `excitement_score` | `FLOAT` | 0–1 how exciting this moment was |
| `audio_url` | `TEXT` | MinIO URL of TTS audio clip (nullable) |
| `generated_at` | `TIMESTAMPTZ` | When generated |
| `model` | `VARCHAR(50)` | LLM used for this line |

### 11.4 LLM Prompt Design

#### System Prompt

```
You are an expert football commentator. Your job is to produce vivid, accurate,
one-sentence commentary for football match events.

Rules:
1. Output ONLY valid JSON: {"commentary": "...one sentence..."}
2. Vary your language — never use the same phrase twice in 10 events
3. Reference player numbers (not names unless provided)
4. Include relevant stats when interesting (xG, speed, possession)
5. Match the requested style: {style}

Style definitions:
- professional: BBC Sport level, measured, factual
- excited: Sky Sports level, energetic, exclamatory  
- tactical: Coaching-grade, focuses on positioning and decisions
- casual: Pub commentary, fun and informal
```

#### User Prompt (per event)

```json
{
  "context": {
    "score": "Team A 1 - 0 Team B",
    "match_minute": 62,
    "possession": "Team A 58% - Team B 42%",
    "momentum": "Team A dominating, 3 shots in last 5 minutes",
    "recent_events": [
      "61:42 — Shot by #7 (Team A), saved",
      "61:55 — Corner to Team A",
      "62:10 — Cross from #11 (Team A)"
    ]
  },
  "current_event": {
    "type": "goal",
    "team": "team_a",
    "player_number": 9,
    "xg": 0.71,
    "distance_m": 11.2,
    "body_part": "right_foot",
    "assist_by": 11,
    "goal_number_for_team": 2
  }
}
```

---

## 12. Service 9 — Report Generation

### 12.1 Overview

Compiles a comprehensive PDF match report using Jinja2 HTML templates rendered to PDF via WeasyPrint. Includes stats tables, heatmap images, shot maps, player leaderboards, annotated frame thumbnails, and AI commentary highlights.

### 12.2 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/matches/{match_id}/report/generate` | JWT | Trigger async PDF generation |
| `GET` | `/api/v1/matches/{match_id}/report/status` | JWT | Poll generation status |
| `GET` | `/api/v1/matches/{match_id}/report/download` | JWT | Download completed PDF |
| `POST` | `/api/v1/matches/{match_id}/report/share` | JWT | Create public share link (24h TTL) |
| `POST` | `/api/v1/matches/{match_id}/report/email` | JWT | Email report to specified address |

### 12.3 Report Contents

```
Page 1:  Cover — Match title, date, venue, final score, thumbnail frame
Page 2:  Executive Summary — Key stats table, xG comparison, possession donut
Page 3:  Match Timeline — Event timeline chart, score progression
Page 4:  Possession Analysis — Time-series chart, zone breakdown bar chart
Page 5:  Shot Analysis — Shot map (pitch diagram), xG table, goal descriptions
Page 6:  Passing Networks — Pass network graph per team, pass accuracy stats
Page 7:  Physical Stats — Distance, speed, sprint comparison table + bar charts
Page 8:  Heatmaps — Side-by-side team heatmaps (pitch overlay)
Page 9:  Formation Analysis — Formation snapshot diagram per team
Page 10: Player Leaderboards — Top 5 per key stat category
Page 11: Team A — Player stats table (all detected players)
Page 12: Team B — Player stats table
Page 13: AI Commentary Highlights — Top 10 moments with commentary text
Page 14: Appendix — Raw event log, methodology notes
```

---

## 13. Service 10 — Tactical Analysis

### 13.1 Overview

Dedicated service for advanced tactical computations including formation detection, pressing intensity (PPDA), defensive line tracking, pass networks, and bird's-eye tactical view generation.

### 13.2 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/matches/{match_id}/tactical/overview` | JWT | Full tactical analysis summary |
| `GET` | `/api/v1/matches/{match_id}/tactical/formations` | JWT | Formation snapshots over time with confidence |
| `GET` | `/api/v1/matches/{match_id}/tactical/pressing` | JWT | PPDA pressing intensity time-series |
| `GET` | `/api/v1/matches/{match_id}/tactical/defensive_line` | JWT | Defensive line height over time |
| `GET` | `/api/v1/matches/{match_id}/tactical/pass_network` | JWT | Pass network graph (nodes = players, edges = passes) |
| `GET` | `/api/v1/matches/{match_id}/tactical/zones` | JWT | Zone control breakdown (pitch thirds × channels) |
| `GET` | `/api/v1/matches/{match_id}/tactical/birdseye/{frame_no}` | JWT | Bird's-eye top-down player positions at frame |
| `GET` | `/api/v1/matches/{match_id}/tactical/birdseye/video` | JWT | Full bird's-eye tactical video (MP4) |
| `GET` | `/api/v1/matches/{match_id}/tactical/offside` | JWT | Offside situation detections |
| `POST` | `/api/v1/matches/{match_id}/tactical/annotations` | JWT | Save coach annotations on tactical view |

### 13.3 Formation Detection Algorithm

```python
def detect_formation(team_positions_m, team='team_a'):
    """
    Input: list of (x, y) positions in metres for all team players
    excluding goalkeeper.
    
    Steps:
    1. Sort players by x-coordinate (attack/defence axis)
    2. Apply k-means with k=3 or k=4 to find defensive/midfield/attacking lines
    3. Count players in each line
    4. Match to nearest known formation from catalogue
    
    Formation catalogue:
    4-4-2, 4-3-3, 4-2-3-1, 3-5-2, 3-4-3,
    5-3-2, 4-5-1, 4-1-4-1, 3-4-2-1, 4-4-1-1
    """
    outfield = [p for p in team_positions_m if not is_goalkeeper(p)]
    outfield_sorted = sorted(outfield, key=lambda p: p.x)
    
    # k-means into defensive (3), midfield (3), attack (1) lines
    kmeans = KMeans(n_clusters=3)
    kmeans.fit([[p.x, p.y] for p in outfield_sorted])
    
    line_counts = Counter(kmeans.labels_)
    formation_str = '-'.join(str(line_counts[i]) for i in range(3))
    
    return match_formation_catalogue(formation_str)
```

### 13.4 PPDA (Passes Allowed Per Defensive Action)

```
PPDA = (Opponent passes in opposition half) / (Defensive actions in opposition half)

Where defensive actions = tackles + interceptions + fouls in opposition half

Low PPDA (< 7)  = High-intensity pressing
Mid PPDA (7–15) = Moderate pressing
High PPDA (>15) = Low pressing / defensive sitting

Computed in 5-minute rolling windows for time-series analysis.
```

---

## 14. Service 11 — Match Chat Assistant

### 14.1 Overview

An LLM-powered conversational interface allowing coaches and analysts to ask natural-language questions about match data. The LLM is given access to all stats endpoints as **MCP tools** and can query live match data to answer questions.

### 14.2 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/matches/{match_id}/chat` | JWT | Send a message; receive AI answer with cited stats |
| `GET` | `/api/v1/matches/{match_id}/chat/history` | JWT | Retrieve chat history |
| `DELETE` | `/api/v1/matches/{match_id}/chat/history` | JWT | Clear chat history |

### 14.3 Example Conversations

```
User: "Which team pressed more in the second half?"
AI:   "Team A pressed significantly harder in the second half with a PPDA
       of 7.2 compared to Team B's 14.8. Team A made 28 defensive actions
       in Team B's half after the 45-minute mark, compared to just 9 by
       Team B in Team A's half."

User: "Who was the fastest player on the pitch?"
AI:   "Player #9 on Team A recorded the highest speed at 34.2 km/h in
       the 73rd minute during a counter-attack run. The second-fastest
       was Player #7 (Team B) at 33.1 km/h."

User: "Was the first goal predictable based on xG?"
AI:   "Yes, it was a relatively high-probability chance. The shot had
       an xG of 0.71, placing it in the top 15% of shots in this match.
       It came from 11.2m with a wide-open angle after the keeper was
       drawn out, and was the 3rd shot in a 4-minute pressing spell."
```

### 14.4 Chat Schema — `chat_messages` Table

| Column | Type | Description |
|---|---|---|
| `id` | `UUID` | Message ID |
| `match_id` | `UUID` | Parent match |
| `user_id` | `UUID` | Sender |
| `role` | `ENUM` | user / assistant |
| `content` | `TEXT` | Message text |
| `tool_calls` | `JSONB` | MCP tool calls made by the LLM |
| `tool_results` | `JSONB` | Stats data retrieved |
| `model` | `VARCHAR(50)` | LLM model used |
| `tokens_used` | `INTEGER` | Total tokens consumed |
| `created_at` | `TIMESTAMPTZ` | Timestamp |

---

## 15. Service 12 — Highlight Extraction

### 15.1 Overview

Automatically identifies and compiles the most exciting moments in a match into a highlight reel. Uses a composite **excitement score** per event to rank and select clips.

### 15.2 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/matches/{match_id}/highlights/generate` | JWT | Trigger async highlight generation |
| `GET` | `/api/v1/matches/{match_id}/highlights` | JWT | List highlight clips ranked by excitement |
| `GET` | `/api/v1/matches/{match_id}/highlights/{highlight_id}` | JWT | Get single highlight clip URL |
| `GET` | `/api/v1/matches/{match_id}/highlights/video` | JWT | Download compiled reel (MP4) |
| `PATCH` | `/api/v1/matches/{match_id}/highlights/{highlight_id}` | JWT | Mark highlight as included/excluded |

### 15.3 Excitement Score Formula

```python
EXCITEMENT_WEIGHTS = {
    'goal':               10.0,
    'shot_on_target':      4.0,
    'shot_off_target':     1.5,
    'save':                4.5,
    'penalty_awarded':     8.0,
    'tackle_won':          2.0,
    'dribble_succeeded':   2.5,
    'header_on_target':    3.0,
    'cross_key':           1.5,
    'sprint_burst':        0.5,
    'long_range_shot':     2.0,   # shot > 25m
    'high_xg_shot':        3.0,   # xG > 0.5
}

def excitement_score(event, context):
    base = EXCITEMENT_WEIGHTS.get(event.type, 0)
    
    # Multipliers
    if context.score_tied:         base *= 1.3
    if context.match_minute > 80:  base *= 1.5   # late game drama
    if context.xg > 0.7:           base *= 1.4   # big chance
    if context.is_counter_attack:  base *= 1.2
    
    return base
```

Each highlight clip = `[event_timestamp - 8s, event_timestamp + 10s]`.

---

## 16. Service 13 — Multi-Camera Fusion

### 16.1 Overview

Accepts 2–4 synchronised camera feeds of the same match. Fuses detections across cameras using shared homography (all cameras map to same field coordinate system), improving coverage and detection accuracy.

### 16.2 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/matches/{match_id}/cameras` | JWT | Add a secondary camera to a match |
| `GET` | `/api/v1/matches/{match_id}/cameras` | JWT | List all camera feeds |
| `DELETE` | `/api/v1/matches/{match_id}/cameras/{cam_id}` | JWT | Remove a camera feed |
| `POST` | `/api/v1/matches/{match_id}/cameras/sync` | JWT | Run automatic sync (detects offset via landmark matching) |
| `GET` | `/api/v1/matches/{match_id}/cameras/{cam_id}/stats` | JWT | Per-camera detection quality stats |

### 16.3 Camera Fusion Algorithm

```
1. Each camera runs full YOLOv8 + ByteTrack independently
2. Each camera has its own homography matrix → all detections mapped to metres
3. For each frame (after sync offset applied):
   a. Collect all detections from all cameras in field coordinates
   b. Cluster detections within 1.5m radius → same physical player
   c. Merge cluster: highest-confidence detection wins for bbox
   d. Average position across cameras for improved accuracy
   e. Output single merged detection set per frame
4. Temporal sync: cross-correlate ball position signals between cameras
   to find offset (typically ±1s)
```

---

## 17. AI/ML Models & Algorithms — Deep Dive

### 17.1 Complete Model Inventory

| # | Model | Task | Framework | Input | Output | Training Data | Target Metric |
|---|---|---|---|---|---|---|---|
| 1 | YOLOv8x (fine-tuned) | Player/ball detection | Ultralytics / PyTorch | 1280×720 BGR frame | Bounding boxes + class + confidence | SoccerNet + custom 50k frames | mAP@0.5 > 0.88 |
| 2 | ByteTrack | Multi-object tracking | supervision + ByteTrack | Frame detections + frame | Detections with persistent track_id | N/A (no training) | MOTA > 0.75 |
| 3 | YOLOv8x-pose | Skeletal pose | Ultralytics / PyTorch | 640×640 player crop | 17 keypoints × [x, y, conf] | COCO pose + football pose | OKS > 0.72 |
| 4 | KMeans (k=3) | Team classification | scikit-learn | HSV player crop colours | Team label | N/A (unsupervised) | Accuracy > 95% |
| 5 | RANSAC Homography | Field calibration | OpenCV | 4+ point correspondences | 3×3 H matrix | N/A (algorithm) | Reprojection error < 2px |
| 6 | Kalman Filter (4-state) | Ball trajectory smoothing | filterpy | Ball detections (x, y) | Smoothed (x, y, vx, vy) | N/A (algorithm) | N/A |
| 7 | Cubic Spline | Ball gap interpolation | SciPy | Detected ball positions | Filled trajectory | N/A (algorithm) | N/A |
| 8 | SlowFast-R50 (fine-tuned) | Action recognition | PyTorchVideo | 16-frame player crop clips | 9-class action probabilities | AVA + SoccerNet Actions | F1 > 0.84 |
| 9 | XGBoost Classifier | Expected goals (xG) | XGBoost | 13 shot features | xG probability 0–1 | StatsBomb + Wyscout (80k shots) | AUC > 0.78 |
| 10 | K-Nearest Neighbours | Formation classification | scikit-learn | 10×(x,y) outfield positions | Formation string | Labelled formation dataset | Accuracy > 85% |
| 11 | PPDA Algorithm | Pressing intensity | NumPy | Event log (passes + def. actions) | PPDA float (rolling 5-min) | N/A (formula) | N/A |
| 12 | Weighted Rolling Sum | Momentum index | NumPy | Event stream with weights | Momentum 0–100 per team | N/A (formula) | N/A |
| 13 | GPT-4o / Llama 3 70B | Match commentary | OpenAI API / llama.cpp | Structured event JSON | Commentary sentence (JSON) | Prompted (no fine-tuning) | Relevance > 4.2/5 |
| 14 | Bark / Coqui TTS | Audio commentary | Hugging Face | Commentary text | MP3 audio clip | Pretrained | MOS > 3.8 |
| 15 | EasyOCR | Jersey number detection | EasyOCR | Player jersey crop | Digit string | Pretrained | Accuracy > 80% on clear jerseys |
| 16 | OSNet (torchreid) | Player re-identification | torchreid | 256×128 player crop | 512-d appearance embedding | Market-1501 + DukeMTMC | Rank-1 > 0.88 |

### 17.2 YOLOv8 Fine-Tuning Details

```yaml
# data.yaml — custom football detection dataset
train: datasets/football/train/images
val: datasets/football/val/images
nc: 5
names: ['player', 'ball', 'referee', 'goalkeeper', 'goalpost']

# Training config
model: yolov8x.pt       # start from COCO pretrained
epochs: 150
imgsz: 1280
batch: 8                # per GPU; use gradient accumulation for effective batch 32
optimizer: AdamW
lr0: 0.001
lrf: 0.01
warmup_epochs: 3
hsv_h: 0.015            # augmentation
hsv_s: 0.7
hsv_v: 0.4
fliplr: 0.5
mosaic: 1.0
mixup: 0.15
copy_paste: 0.3
```

### 17.3 SlowFast Action Recognition

```python
# Input preparation
def prepare_clip(frames, bbox, clip_len=16):
    """
    Extract a temporal clip centred on current frame.
    Crop to player bounding box + 20% padding.
    Resize to 224×224.
    Normalise using ImageNet mean/std.
    """
    clip = []
    for f in frames:  # 16 frames
        crop = f[y1-pad:y2+pad, x1-pad:x2+pad]
        resized = cv2.resize(crop, (224, 224))
        clip.append(resized)
    
    # SlowFast needs slow + fast pathways
    slow = clip[::4]   # every 4th frame — 4 frames for slow pathway
    fast = clip        # all 16 frames for fast pathway
    
    return torch.tensor(slow), torch.tensor(fast)
```

### 17.4 Momentum Index

```
Momentum score = weighted exponential moving average of events

Event weights:
  goal             = +20 (for scoring team), -15 (for conceding team)
  shot_on_target   = +5
  shot_off_target  = +2
  save             = +8
  tackle_won       = +3
  key_pass         = +4
  possession gain  = +1
  possession loss  = -1

EMA decay: α = 0.05 per second (recent events matter more)
Score clamped to [0, 100] and normalised so team_a + team_b = 100
```

### 17.5 Offside Detection

```python
def detect_offside(team_positions_m, ball_position_m, attacking_team):
    """
    Offside line = second-last defender position (x coordinate)
    A player is offside if:
    1. They are in the opponent's half
    2. They are ahead of the ball
    3. They are ahead of the second-last defender
    4. They are actively receiving or seeking the ball
    """
    defenders = [p for p in team_positions_m 
                 if p.team != attacking_team and p.class_label == 'player']
    
    # Sort defenders by distance to own goal (x coordinate, adjusted for direction)
    defenders_sorted = sorted(defenders, key=lambda p: p.x)
    
    # Second-last defender (last is typically goalkeeper)
    offside_line_x = defenders_sorted[1].x if len(defenders_sorted) >= 2 else 0
    
    # Check each attacker
    for attacker in attacking_players:
        if (attacker.x > offside_line_x and 
            attacker.x > ball_position_m.x and
            attacker.x > 52.5):  # past halfway line
            return OffsideEvent(player=attacker, line_x=offside_line_x)
```

---

## 18. Database Schema — Full Detail

### 18.1 PostgreSQL Tables — Complete DDL

```sql
-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector for ReID embeddings

-- ENUM types
CREATE TYPE user_plan AS ENUM ('free', 'pro', 'enterprise');
CREATE TYPE match_status AS ENUM ('queued', 'processing', 'completed', 'failed', 'cancelled');
CREATE TYPE team_label AS ENUM ('team_a', 'team_b', 'referee', 'unknown');
CREATE TYPE event_type AS ENUM (
    'goal', 'shot', 'pass', 'tackle', 'foul', 'corner', 'offside',
    'save', 'dribble', 'interception', 'clearance', 'free_kick',
    'cross', 'header', 'penalty_awarded', 'yellow_card', 'red_card'
);
CREATE TYPE event_outcome AS ENUM (
    'success', 'failure', 'on_target', 'off_target', 'blocked',
    'saved', 'wide', 'over', 'goal', 'miss'
);
CREATE TYPE body_part AS ENUM ('right_foot', 'left_foot', 'head', 'other');
CREATE TYPE action_label AS ENUM (
    'standing', 'jogging', 'sprinting', 'passing', 'receiving',
    'shooting', 'heading', 'tackling', 'dribbling', 'saving', 'celebrating'
);
CREATE TYPE commentary_style AS ENUM ('professional', 'excited', 'tactical', 'casual');
CREATE TYPE report_status AS ENUM ('queued', 'generating', 'completed', 'failed');

-- ================================================
-- USERS
-- ================================================
CREATE TABLE users (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    email            VARCHAR(255) UNIQUE NOT NULL,
    password_hash    VARCHAR(60)  NOT NULL,
    full_name        VARCHAR(100),
    organisation     VARCHAR(100),
    plan             user_plan    NOT NULL DEFAULT 'free',
    is_active        BOOLEAN      NOT NULL DEFAULT true,
    email_verified   BOOLEAN      NOT NULL DEFAULT false,
    storage_quota_bytes BIGINT   NOT NULL DEFAULT 10737418240, -- 10 GB
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_login_at    TIMESTAMPTZ
);
CREATE INDEX idx_users_email ON users(email);

-- ================================================
-- API KEYS
-- ================================================
CREATE TABLE api_keys (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    label        VARCHAR(100) NOT NULL,
    key_hash     VARCHAR(64)  NOT NULL UNIQUE,
    key_prefix   VARCHAR(20)  NOT NULL,
    is_active    BOOLEAN      NOT NULL DEFAULT true,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    expires_at   TIMESTAMPTZ
);
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- ================================================
-- MATCHES
-- ================================================
CREATE TABLE matches (
    id                     UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title                  VARCHAR(200) NOT NULL,
    team_a_name            VARCHAR(100) NOT NULL DEFAULT 'Team A',
    team_b_name            VARCHAR(100) NOT NULL DEFAULT 'Team B',
    team_a_jersey_color    VARCHAR(7),
    team_b_jersey_color    VARCHAR(7),
    match_date             DATE,
    venue                  VARCHAR(200),
    competition            VARCHAR(200),
    video_url              TEXT         NOT NULL,
    annotated_url          TEXT,
    thumbnail_url          TEXT,
    file_size_bytes        BIGINT       NOT NULL,
    duration_seconds       FLOAT,
    fps                    FLOAT        NOT NULL,
    resolution_w           INTEGER      NOT NULL,
    resolution_h           INTEGER      NOT NULL,
    total_frames           INTEGER,
    processed_frames       INTEGER      NOT NULL DEFAULT 0,
    status                 match_status NOT NULL DEFAULT 'queued',
    failure_reason         TEXT,
    is_live                BOOLEAN      NOT NULL DEFAULT false,
    analysis_config        JSONB,
    homography_matrix      FLOAT[],     -- 9-element flat array
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
    processing_started_at  TIMESTAMPTZ,
    completed_at           TIMESTAMPTZ
);
CREATE INDEX idx_matches_user_id ON matches(user_id);
CREATE INDEX idx_matches_status ON matches(status);
CREATE INDEX idx_matches_created_at ON matches(created_at DESC);

-- ================================================
-- FRAMES (partitioned by match_id for performance)
-- ================================================
CREATE TABLE frames (
    id                  UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id            UUID    NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    frame_number        INTEGER NOT NULL,
    timestamp_ms        FLOAT   NOT NULL,
    detections          JSONB   NOT NULL DEFAULT '[]',
    ball_position_px    POINT,
    ball_position_m     POINT,
    ball_velocity_mps   FLOAT,
    ball_confidence     FLOAT,
    homography_matrix   FLOAT[],
    processing_time_ms  FLOAT,
    UNIQUE(match_id, frame_number)
);
CREATE INDEX idx_frames_match_frame ON frames(match_id, frame_number);
CREATE INDEX idx_frames_timestamp ON frames(match_id, timestamp_ms);

-- ================================================
-- MATCH STATS
-- ================================================
CREATE TABLE match_stats (
    id                        UUID  PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id                  UUID  NOT NULL UNIQUE REFERENCES matches(id) ON DELETE CASCADE,
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_frame_number         INTEGER,
    -- Possession
    possession_a_pct          FLOAT,
    possession_b_pct          FLOAT,
    possession_a_seconds      FLOAT,
    possession_b_seconds      FLOAT,
    possession_a_own_half     FLOAT,
    possession_a_opp_half     FLOAT,
    possession_b_own_half     FLOAT,
    possession_b_opp_half     FLOAT,
    -- Scoring
    goals_a                   INTEGER NOT NULL DEFAULT 0,
    goals_b                   INTEGER NOT NULL DEFAULT 0,
    shots_a                   INTEGER NOT NULL DEFAULT 0,
    shots_on_target_a         INTEGER NOT NULL DEFAULT 0,
    shots_b                   INTEGER NOT NULL DEFAULT 0,
    shots_on_target_b         INTEGER NOT NULL DEFAULT 0,
    xg_a                      FLOAT   NOT NULL DEFAULT 0,
    xg_b                      FLOAT   NOT NULL DEFAULT 0,
    -- Passing
    passes_attempted_a        INTEGER,
    passes_completed_a        INTEGER,
    pass_accuracy_a           FLOAT,
    passes_attempted_b        INTEGER,
    passes_completed_b        INTEGER,
    pass_accuracy_b           FLOAT,
    -- Defensive
    tackles_a                 INTEGER,
    tackles_won_a             INTEGER,
    tackles_b                 INTEGER,
    tackles_won_b             INTEGER,
    fouls_a                   INTEGER,
    fouls_b                   INTEGER,
    -- Set pieces
    corners_a                 INTEGER,
    corners_b                 INTEGER,
    offsides_a                INTEGER,
    offsides_b                INTEGER,
    -- Physical
    total_distance_a_km       FLOAT,
    total_distance_b_km       FLOAT,
    avg_speed_a_kmh           FLOAT,
    avg_speed_b_kmh           FLOAT,
    max_speed_a_kmh           FLOAT,
    max_speed_b_kmh           FLOAT,
    sprints_a                 INTEGER,
    sprints_b                 INTEGER,
    -- Tactical
    formation_a               VARCHAR(10),
    formation_b               VARCHAR(10),
    pressing_intensity_a      FLOAT,
    pressing_intensity_b      FLOAT,
    defensive_line_height_a   FLOAT,
    defensive_line_height_b   FLOAT,
    momentum_a                FLOAT,
    momentum_b                FLOAT
);

-- ================================================
-- EVENTS
-- ================================================
CREATE TABLE events (
    id                          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id                    UUID         NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    type                        event_type   NOT NULL,
    timestamp_ms                FLOAT        NOT NULL,
    match_minute                INTEGER,
    team                        team_label   NOT NULL,
    player_track_id             INTEGER,
    secondary_player_track_id   INTEGER,
    location_px                 POINT,
    location_m                  POINT,
    outcome                     event_outcome,
    xg_value                    FLOAT,
    pass_distance_m             FLOAT,
    shot_distance_m             FLOAT,
    shot_angle_deg              FLOAT,
    body_part                   body_part,
    confidence                  FLOAT        NOT NULL,
    frame_number                INTEGER
);
CREATE INDEX idx_events_match_id ON events(match_id);
CREATE INDEX idx_events_type ON events(match_id, type);
CREATE INDEX idx_events_timestamp ON events(match_id, timestamp_ms);

-- ================================================
-- PLAYER STATS
-- ================================================
CREATE TABLE player_stats (
    id                      UUID       PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id                UUID       NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    track_id                INTEGER    NOT NULL,
    team                    team_label NOT NULL,
    player_name             VARCHAR(100),
    jersey_number           SMALLINT,
    dominant_jersey_color   VARCHAR(7),
    first_seen_frame        INTEGER,
    last_seen_frame         INTEGER,
    frames_detected         INTEGER    NOT NULL DEFAULT 0,
    time_on_pitch_seconds   FLOAT,
    -- Physical
    total_distance_km       FLOAT,
    avg_speed_kmh           FLOAT,
    max_speed_kmh           FLOAT,
    max_speed_timestamp_ms  FLOAT,
    sprints_count           INTEGER,
    high_intensity_runs     INTEGER,
    -- Ball skills
    touches                 INTEGER,
    passes_attempted        INTEGER,
    passes_completed        INTEGER,
    pass_accuracy           FLOAT,
    shots                   INTEGER,
    shots_on_target         INTEGER,
    goals                   INTEGER,
    assists                 INTEGER,
    xg_total                FLOAT,
    tackles_won             INTEGER,
    interceptions           INTEGER,
    -- Spatial
    avg_position_x          FLOAT,
    avg_position_y          FLOAT,
    heatmap                 FLOAT[],   -- 32×52 = 1664 floats
    reid_embedding          VECTOR(512), -- pgvector for cross-match identity
    UNIQUE(match_id, track_id)
);
CREATE INDEX idx_player_stats_match ON player_stats(match_id);
CREATE INDEX idx_player_stats_reid ON player_stats USING ivfflat (reid_embedding vector_cosine_ops);

-- ================================================
-- COMMENTARY
-- ================================================
CREATE TABLE commentary (
    id               UUID               PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id         UUID               NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    event_id         UUID               REFERENCES events(id) ON DELETE SET NULL,
    frame_number     INTEGER,
    timestamp_ms     FLOAT,
    text             TEXT               NOT NULL,
    style            commentary_style   NOT NULL DEFAULT 'professional',
    language         VARCHAR(5)         NOT NULL DEFAULT 'en',
    excitement_score FLOAT,
    audio_url        TEXT,
    generated_at     TIMESTAMPTZ        NOT NULL DEFAULT now(),
    model            VARCHAR(50)
);

-- ================================================
-- REPORTS
-- ================================================
CREATE TABLE reports (
    id           UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id     UUID          NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    url          TEXT,
    status       report_status NOT NULL DEFAULT 'queued',
    generated_at TIMESTAMPTZ,
    file_size_bytes INTEGER,
    share_token  VARCHAR(64),
    share_expires_at TIMESTAMPTZ
);

-- ================================================
-- CAMERAS
-- ================================================
CREATE TABLE cameras (
    id              UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id        UUID    NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    label           VARCHAR(100),
    video_url       TEXT    NOT NULL,
    homography_matrix FLOAT[],
    sync_offset_ms  FLOAT   NOT NULL DEFAULT 0,
    is_primary      BOOLEAN NOT NULL DEFAULT false,
    detection_quality FLOAT,
    coverage_pct    FLOAT
);

-- ================================================
-- CHAT MESSAGES
-- ================================================
CREATE TABLE chat_messages (
    id           UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id     UUID    NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    user_id      UUID    NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role         VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT    NOT NULL,
    tool_calls   JSONB,
    tool_results JSONB,
    model        VARCHAR(50),
    tokens_used  INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ================================================
-- POSSESSION TIME SERIES (high-write; separate from match_stats)
-- ================================================
CREATE TABLE possession_timeseries (
    id         UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id   UUID    NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    second     INTEGER NOT NULL,
    team_a_pct FLOAT   NOT NULL,
    team_b_pct FLOAT   NOT NULL,
    UNIQUE(match_id, second)
);
CREATE INDEX idx_possession_match ON possession_timeseries(match_id);
```

### 18.2 Redis Key Schema

| Key Pattern | Type | TTL | Contents |
|---|---|---|---|
| `match:{id}:stats` | Hash | 1 day | Latest stats snapshot (all fields) |
| `match:{id}:frames:latest` | String | 5 sec | Most recent frame JSON |
| `match:{id}:ball:trajectory` | List | 1 day | Last 300 ball positions (x,y,t) |
| `session:{id}:state` | Hash | 24 hrs | Live session metadata |
| `session:{id}:frames_received` | Integer | 24 hrs | Frame count for live session |
| `queue:inference` | Stream | — | CV job tasks (Redis Stream) |
| `queue:stats` | Stream | — | Stats tasks (Redis Stream) |
| `queue:reports` | Stream | — | Report generation tasks |
| `pubsub:match:{id}` | Pub/Sub | — | Real-time stats delivery channel |
| `ratelimit:user:{id}` | Integer | 60 sec | Request count for rate limiting |
| `jwt:blacklist:{jti}` | String | 30 days | Revoked JWT IDs |
| `processing:match:{id}:progress` | Float | 7 days | 0.0–1.0 processing progress |

---

## 19. API Design Conventions

### 19.1 Base URL

```
Production:  https://api.footballiq.com/api/v1
Staging:     https://api-staging.footballiq.com/api/v1
Local:       http://localhost:8000/api/v1
```

### 19.2 Request Headers

```http
Authorization: Bearer <JWT>
X-API-Key: fiq_live_xxxxxxxxxxxxxxxxxxxxxxxx
Content-Type: application/json
Accept: application/json
X-Request-ID: <client-generated-uuid>   (optional; echoed in response)
```

### 19.3 Pagination Convention

All list endpoints follow cursor-based pagination:

```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 143,
    "has_next": true,
    "has_prev": false
  }
}
```

### 19.4 Filtering & Sorting

```
GET /api/v1/matches?status=completed&sort=created_at&order=desc&from_date=2025-01-01
GET /api/v1/matches/{id}/stats/events?type=shot&outcome=on_target&from_ms=0&to_ms=2700000
GET /api/v1/matches/{id}/players?team=team_a&sort=total_distance_km&order=desc
```

---

## 20. Error Handling & Response Formats

### 20.1 Success Response

```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "request_id": "uuid",
    "timestamp": "ISO 8601",
    "version": "1.0"
  }
}
```

### 20.2 Error Response

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable description",
    "details": [
      { "field": "file", "issue": "File type not supported. Accepted: mp4, mkv, avi, mov" }
    ],
    "request_id": "uuid",
    "timestamp": "ISO 8601"
  }
}
```

### 20.3 Error Code Reference

| HTTP Status | Code | Meaning |
|---|---|---|
| `400` | `VALIDATION_ERROR` | Request body / query param validation failure |
| `400` | `INVALID_FILE_FORMAT` | Unsupported video file format |
| `400` | `FILE_TOO_LARGE` | Exceeds plan limit (free: 2GB, pro: 10GB) |
| `401` | `UNAUTHORIZED` | Missing or invalid auth token |
| `401` | `TOKEN_EXPIRED` | JWT has expired; refresh needed |
| `403` | `FORBIDDEN` | Authenticated but not allowed (wrong plan) |
| `403` | `QUOTA_EXCEEDED` | Storage or API rate limit exceeded |
| `404` | `NOT_FOUND` | Resource does not exist or belongs to another user |
| `409` | `CONFLICT` | Duplicate resource (e.g. same email on register) |
| `422` | `PROCESSING_NOT_COMPLETE` | Stats requested but match still processing |
| `429` | `RATE_LIMITED` | Too many requests; retry after header provided |
| `500` | `INFERENCE_ERROR` | CV pipeline error during frame processing |
| `500` | `INTERNAL_ERROR` | Unexpected server error |
| `503` | `GPU_UNAVAILABLE` | No GPU worker available; job queued |

---

## 21. Authentication & Security

### 21.1 Authentication Flow

```
Login → JWT (access=30min) + Refresh (30 days)
       ↓
Expired access token → POST /auth/refresh with refresh token → new access token
       ↓
Logout → refresh token added to Redis blacklist (TTL = remaining lifetime)
```

### 21.2 API Key Format

```
fiq_{env}_{32-char-random}
Examples:
  fiq_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
  fiq_test_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

- env: live | test
- Stored as SHA-256 hash only; shown in plaintext once on creation
- Rate limited at 500 req/min (pro) / 2000 req/min (enterprise)
```

### 21.3 Security Measures

| Measure | Implementation |
|---|---|
| Password hashing | bcrypt with cost factor 12 |
| JWT signing | RS256 (asymmetric); private key in HSM/env |
| Token blacklisting | JWT `jti` stored in Redis on logout |
| Input validation | Pydantic v2 strict mode on all endpoints |
| SQL injection | SQLAlchemy ORM; parameterised queries only |
| File upload safety | Magic byte validation; MIME type check; ClamAV scan |
| Rate limiting | Redis-based sliding window; per user per endpoint |
| CORS | Configurable origins; credentials only on trusted origins |
| TLS | TLS 1.3 minimum; HSTS header enforced |
| Secrets | All secrets in environment variables; never in code |
| API key storage | SHA-256 hashed; never stored in plaintext |

---

## 22. Performance & Scalability

### 22.1 Inference Performance Targets

| Metric | Target | Hardware |
|---|---|---|
| Single frame (detection only) | < 50ms | RTX 3090 |
| Single frame (full pipeline) | < 150ms | RTX 3090 |
| Live stream end-to-end latency | < 200ms | RTX 3090 |
| Throughput (uploaded video) | 15–20 FPS processing | RTX 3090 |
| Max concurrent live sessions | 4 | Single RTX 3090 |

### 22.2 Database Performance

```sql
-- Critical indices
CREATE INDEX idx_frames_match_frame ON frames(match_id, frame_number);  -- frame lookup
CREATE INDEX idx_events_match_type ON events(match_id, type);           -- event filtering
CREATE INDEX idx_player_stats_match ON player_stats(match_id);          -- player aggregation
CREATE INDEX idx_possession_match ON possession_timeseries(match_id);   -- time-series

-- Partitioning: frames table partitioned by match_id using PARTITION BY HASH
-- Each partition covers ~10 matches → keeps table scan sizes manageable
```

### 22.3 Caching Strategy

| Data | Cache | TTL | Strategy |
|---|---|---|---|
| Latest match stats | Redis Hash | 5 sec (live), 1 day (done) | Write-through from Stats Engine |
| User profile | Redis | 5 min | Cache-aside on auth middleware |
| Player stats (completed match) | Redis | 1 day | Populated on match completion |
| API rate limit counters | Redis | 60 sec | Sliding window counter |
| JWT blacklist | Redis | JWT remaining lifetime | Write on logout |

### 22.4 Horizontal Scaling

```yaml
# Kubernetes HPA config (example)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-worker
  minReplicas: 1
  maxReplicas: 8
  metrics:
  - type: External
    external:
      metric:
        name: redis_stream_queue_length
        selector:
          matchLabels:
            stream: inference
      target:
        type: AverageValue
        averageValue: "5"    # scale up when queue depth > 5 per pod
```

---

## 23. Infrastructure & Deployment

### 23.1 Docker Compose (Development)

```yaml
version: "3.9"
services:
  api:
    build: ./services/api
    ports: ["8000:8000"]
    environment:
      DATABASE_URL: postgresql://postgres:pass@db:5432/footballiq
      REDIS_URL: redis://redis:6379
      MINIO_ENDPOINT: minio:9000
      JWT_PRIVATE_KEY_FILE: /secrets/jwt_rs256.pem
    depends_on: [db, redis, minio]

  inference-worker:
    build: ./services/inference
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    environment:
      REDIS_URL: redis://redis:6379
      DATABASE_URL: postgresql://postgres:pass@db:5432/footballiq
      MINIO_ENDPOINT: minio:9000
    depends_on: [redis, db]

  stats-engine:
    build: ./services/stats
    environment:
      REDIS_URL: redis://redis:6379
      DATABASE_URL: postgresql://postgres:pass@db:5432/footballiq
    depends_on: [redis, db]

  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: footballiq
      POSTGRES_PASSWORD: pass
    volumes: [postgres_data:/var/lib/postgresql/data]
    ports: ["5432:5432"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports: ["9000:9000", "9001:9001"]
    volumes: [minio_data:/data]

  celery-worker:
    build: ./services/api
    command: celery -A app.celery worker -Q reports,highlights --loglevel=info
    depends_on: [redis, db]

volumes:
  postgres_data:
  minio_data:
```

### 23.2 Environment Variables

```bash
# API Service
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/footballiq
REDIS_URL=redis://host:6379/0
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
MINIO_BUCKET=footballiq
JWT_PRIVATE_KEY=...     # RS256 PEM
JWT_PUBLIC_KEY=...      # RS256 PEM
JWT_ACCESS_EXPIRE_MINUTES=30
JWT_REFRESH_EXPIRE_DAYS=30
OPENAI_API_KEY=...      # for GPT-4o commentary
CORS_ORIGINS=https://app.footballiq.com

# Inference Worker
MODEL_DIR=/models
DETECTION_MODEL=yolov8x_football.pt
POSE_MODEL=yolov8x-pose.pt
ACTION_MODEL=slowfast_r50_football.pt
CUDA_VISIBLE_DEVICES=0
FRAME_BATCH_SIZE=1

# Stats Engine
POSSESSION_THRESHOLD_M=2.0
SPRINT_THRESHOLD_KMH=25.0
HIGH_INTENSITY_THRESHOLD_KMH=19.8
STATS_WRITE_INTERVAL_SECONDS=1.0
```

---

## 24. Non-Functional Requirements

| Category | Requirement | Target |
|---|---|---|
| **Latency** | Live frame processing end-to-end | < 200ms |
| **Latency** | REST stats API (P99) | < 400ms |
| **Latency** | WebSocket stat delivery | < 50ms after stats engine update |
| **Throughput** | Uploaded video processing speed | ≥ 15 FPS on RTX 3090 |
| **Throughput** | Concurrent live sessions per GPU | 4 sessions |
| **Throughput** | API requests/sec (single pod) | 200 req/s |
| **Availability** | API uptime SLA | 99.5% monthly |
| **Availability** | Data durability (MinIO) | 99.999999% (erasure coding) |
| **Scalability** | Inference workers (Kubernetes) | 1–32 GPU pods auto-scaled |
| **Storage** | Max upload per request | 10 GB (pro plan) |
| **Storage** | Raw frame retention | 90 days |
| **Storage** | Stats + events retention | 365 days |
| **Security** | Authentication | JWT RS256 + API Key |
| **Security** | Transport | TLS 1.3 minimum |
| **Security** | Password hashing | bcrypt cost=12 |
| **Compliance** | GDPR | User data deletion on request within 30 days |
| **Observability** | Metrics | Prometheus + Grafana dashboard |
| **Observability** | Logging | Structured JSON logs → ELK stack |
| **Observability** | Tracing | OpenTelemetry → Jaeger |

---

## 25. Technology Stack — Full Reference

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **API Framework** | FastAPI | 0.111+ | Async REST + WebSocket + auto OpenAPI docs |
| **ASGI Server** | uvicorn | 0.29+ | Production ASGI server with uvloop |
| **Data Validation** | Pydantic v2 | 2.7+ | Request/response validation; strict mode |
| **ORM** | SQLAlchemy | 2.0+ | Async ORM; raw SQL via text() for complex queries |
| **DB Migrations** | Alembic | 1.13+ | Schema versioning |
| **PostgreSQL** | pgvector/pgvector:pg16 | PG 16 | Primary relational DB + pgvector for ReID |
| **Redis** | redis:7-alpine | 7.2+ | Cache, queue, pub/sub, rate limiting |
| **Object Storage** | MinIO | RELEASE.2025 | S3-compatible; videos, reports, frames |
| **Task Queue** | Celery | 5.3+ | Async background jobs (reports, highlights) |
| **Object Detection** | Ultralytics YOLOv8 | 8.2+ | Player, ball, referee detection |
| **Tracking** | supervision (Roboflow) | 0.20+ | ByteTrack wrapper + annotation utilities |
| **Video I/O** | OpenCV-Python | 4.9+ | Frame reading, annotation drawing, video write |
| **Pose Estimation** | YOLOv8-pose | — | 17-keypoint skeleton per player |
| **Action Recognition** | PyTorchVideo | 0.1.5+ | SlowFast for action classification |
| **ML / Stats** | NumPy | 1.26+ | Core numerical computing |
| **ML / Stats** | SciPy | 1.12+ | Kalman filter, spline interpolation, signal processing |
| **ML / Stats** | scikit-learn | 1.4+ | KMeans, KNN, logistic regression, preprocessing |
| **xG Model** | XGBoost | 2.0+ | Expected goals classifier |
| **Kalman Filter** | filterpy | 1.4+ | Ball tracking Kalman filter |
| **Player ReID** | torchreid | 1.4+ | OSNet appearance embeddings |
| **OCR** | EasyOCR | 1.7+ | Jersey number detection |
| **LLM Commentary** | openai | 1.30+ | GPT-4o API for commentary |
| **LLM (local)** | llama-cpp-python | 0.2+ | Llama 3 70B for on-premise commentary |
| **TTS Audio** | Bark / Coqui TTS | — | Text-to-speech audio commentary |
| **PDF Reports** | WeasyPrint | 62+ | HTML → PDF report rendering |
| **Report Templates** | Jinja2 | 3.1+ | HTML templates for report pages |
| **Report Charts** | matplotlib | 3.8+ | Heatmap, shot map, timeline chart images |
| **Auth** | python-jose | 3.3+ | JWT RS256 encode/decode |
| **Auth** | passlib | 1.7+ | bcrypt password hashing |
| **Containers** | Docker | 26+ | Service containerisation |
| **Orchestration** | Kubernetes | 1.29+ | Production orchestration + HPA autoscaling |
| **Monitoring** | Prometheus + Grafana | — | Metrics collection and dashboards |
| **Logging** | structlog | 24+ | Structured JSON logging |
| **Tracing** | OpenTelemetry | 1.24+ | Distributed tracing |
| **Testing** | pytest + pytest-asyncio | — | Unit + integration testing |
| **API Testing** | httpx | 0.27+ | Async HTTP client for tests |
| **CI/CD** | GitHub Actions | — | Build, test, push, deploy pipeline |

---

## 26. Implementation Roadmap

### Phase 1 — Foundation (Weeks 1–3)

**Goal:** Working API skeleton, database, storage, auth, and basic video upload.

| Task | Owner | Estimate |
|---|---|---|
| FastAPI project structure, settings, logging, error handlers | Backend | 2d |
| PostgreSQL schema + Alembic migrations | Backend | 1d |
| Auth service: register, login, JWT, refresh, logout | Backend | 3d |
| MinIO integration: bucket setup, upload/download helpers | Backend | 1d |
| Match ingestion: upload endpoint, FFprobe metadata extraction | Backend | 2d |
| Redis setup: queue, cache, pub/sub abstractions | Backend | 1d |
| Docker Compose: all services running locally | DevOps | 1d |
| Basic test suite: auth + ingestion endpoints | QA | 2d |

### Phase 2 — Core CV Pipeline (Weeks 4–7)

**Goal:** YOLOv8 + ByteTrack + team classification running on uploaded videos.

| Task | Owner | Estimate |
|---|---|---|
| YOLOv8x fine-tuning on football dataset | ML | 5d |
| Inference Worker: Redis Streams job consumer | ML | 2d |
| OpenCV frame reader + letterbox preprocessing | ML | 1d |
| YOLOv8 detection integration | ML | 2d |
| ByteTrack tracking integration (supervision) | ML | 2d |
| KMeans team classification + temporal smoothing | ML | 3d |
| Frame schema: write detections to PostgreSQL | ML | 1d |
| Homography: manual calibration mode | ML | 2d |
| Basic stats: possession + speed calculation | ML | 2d |
| Processing progress tracking (WebSocket + polling) | Backend | 1d |

### Phase 3 — Statistics Engine (Weeks 8–11)

**Goal:** Full stats engine with all 30+ statistics, events, and WebSocket delivery.

| Task | Owner | Estimate |
|---|---|---|
| Stats Engine service structure + Redis Streams consumer | Backend | 2d |
| Goal detection algorithm | ML | 2d |
| Pass detection (possession transitions) | ML | 2d |
| Shot detection (ball velocity + trajectory to goal) | ML | 3d |
| Tackle / foul detection (action model integration) | ML | 3d |
| Corner, offside, free kick detection | ML | 3d |
| Full match_stats table population | Backend | 2d |
| WebSocket hub: Redis pub/sub → connected clients | Backend | 2d |
| Events API endpoints | Backend | 1d |
| Stats API endpoints (all 18 endpoints) | Backend | 2d |
| Possession time-series table + API | Backend | 1d |

### Phase 4 — Advanced CV (Weeks 12–15)

**Goal:** Pose estimation, action recognition, homography auto-calibration, xG.

| Task | Owner | Estimate |
|---|---|---|
| YOLOv8-pose integration + keypoint schema | ML | 2d |
| SlowFast fine-tuning on SoccerNet Actions | ML | 5d |
| Action recognition integration into inference pipeline | ML | 2d |
| Kalman filter + cubic spline ball tracking | ML | 2d |
| Auto-homography: field line detection | ML | 4d |
| xG model: feature engineering + XGBoost training | ML | 3d |
| xG API endpoints + timeline | Backend | 1d |
| Jersey number OCR (EasyOCR) | ML | 2d |
| Player ReID embeddings (torchreid + pgvector) | ML | 3d |

### Phase 5 — Tactical & Player Analytics (Weeks 16–18)

**Goal:** Formation detection, PPDA, heatmaps, pass networks, player deep stats.

| Task | Owner | Estimate |
|---|---|---|
| Formation detection (KNN classifier) | ML | 3d |
| PPDA pressing intensity calculation | ML | 2d |
| Defensive line height tracking | ML | 2d |
| Heatmap generation (all players + per-player) | ML | 2d |
| Pass network graph API | Backend | 2d |
| Momentum index accumulator | ML | 1d |
| Player stats aggregation (all 40 fields) | Backend | 3d |
| Player comparison + leaderboard APIs | Backend | 2d |
| Tactical bird's-eye view generation | ML | 3d |
| Offside detection algorithm | ML | 2d |

### Phase 6 — AI Features (Weeks 19–21)

**Goal:** Commentary, chat assistant, highlight extraction, PDF reports.

| Task | Owner | Estimate |
|---|---|---|
| AI commentary: LLM integration + prompt design | ML | 3d |
| Commentary WebSocket streaming | Backend | 1d |
| TTS audio commentary (Bark) | ML | 2d |
| Match chat assistant (MCP tools + LLM) | ML | 4d |
| Highlight extraction: excitement scoring | ML | 3d |
| Highlight reel compiler (FFmpeg) | ML | 2d |
| PDF report service: Jinja2 templates + WeasyPrint | Backend | 4d |
| Report charts: heatmap, shot map, timeline images | Backend | 3d |
| Report share link + email delivery | Backend | 1d |

### Phase 7 — Multi-Camera & Polish (Weeks 22–24)

| Task | Owner | Estimate |
|---|---|---|
| Multi-camera fusion algorithm | ML | 4d |
| Camera sync (cross-correlation) | ML | 2d |
| Camera management API | Backend | 2d |
| Performance profiling + GPU optimisation | ML | 3d |
| Load testing (k6 or Locust) | QA | 2d |
| Kubernetes Helm chart | DevOps | 3d |
| Prometheus metrics + Grafana dashboards | DevOps | 2d |
| OpenTelemetry tracing integration | DevOps | 2d |
| Security audit (OWASP) | Security | 3d |
| API documentation review + Postman collection | Backend | 2d |
| Public beta launch | All | — |

---

## 27. Appendix — Stat Calculation Formulas

### Pass Accuracy
```
pass_accuracy = (passes_completed / passes_attempted) × 100
```

### Distance Covered (per player, per frame interval)
```
dist_frame = sqrt((x2 - x1)² + (y2 - y1)²)   [metres]
Smoothed with 5-frame rolling average to reduce tracking noise
total_distance = Σ dist_frame × frame_time_seconds
```

### Sprint Detection
```
A sprint is a continuous period where speed > 25 km/h
  AND duration > 1 second
  AND followed by speed < 20 km/h for > 2 seconds
sprints_count = number of such periods
```

### Heatmap Generation
```python
# 32×52 grid = pitch height × pitch width (metres/2.125)
def build_heatmap(positions_m, pitch_w=105, pitch_h=68, bins=(52, 32)):
    x_coords = [p.x for p in positions_m]
    y_coords = [p.y for p in positions_m]
    heatmap, _, _ = np.histogram2d(
        x_coords, y_coords,
        bins=bins,
        range=[[0, pitch_w], [0, pitch_h]]
    )
    # Gaussian blur for smooth appearance
    heatmap = gaussian_filter(heatmap, sigma=1.5)
    # Normalise to 0–1
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap.flatten().tolist()
```

### xG — Distance and Angle Features
```python
def shot_features(ball_pos_m, goal_centre_m=(105, 34)):
    dx = goal_centre_m[0] - ball_pos_m.x
    dy = goal_centre_m[1] - ball_pos_m.y
    distance_m = sqrt(dx**2 + dy**2)
    
    # Angle subtended by goal (7.32m wide) from shot position
    goal_left_m  = (105, 34 - 3.66)
    goal_right_m = (105, 34 + 3.66)
    
    a = atan2(goal_left_m[1] - ball_pos_m.y, goal_left_m[0] - ball_pos_m.x)
    b = atan2(goal_right_m[1] - ball_pos_m.y, goal_right_m[0] - ball_pos_m.x)
    
    angle_deg = abs(degrees(a - b))
    return distance_m, angle_deg
```

### Momentum Index
```python
def update_momentum(events_last_5min, team):
    score = 0
    weights = {
        'goal': 20, 'shot_on_target': 5, 'shot_off_target': 2,
        'save': 8, 'tackle_won': 3, 'key_pass': 4
    }
    for event in events_last_5min:
        if event.team == team:
            score += weights.get(event.type, 0)
        else:
            score -= weights.get(event.type, 0) * 0.5
    
    # Normalise: EMA of raw score, clamped 0–100
    momentum_raw[team] = 0.9 * momentum_raw[team] + 0.1 * score
    return max(0, min(100, 50 + momentum_raw[team]))
```

### Possession by Zone
```python
# Pitch divided into 9 zones: 3 thirds × 3 channels
# Thirds: defensive (0–35m), middle (35–70m), attacking (70–105m)
# Channels: left (0–22.7m), central (22.7–45.3m), right (45.3–68m)

def classify_zone(pos_m):
    x, y = pos_m.x, pos_m.y
    third = 'def' if x < 35 else 'mid' if x < 70 else 'att'
    channel = 'left' if y < 22.7 else 'right' if y > 45.3 else 'central'
    return f"{third}_{channel}"
```

---

*End of FootballIQ PRD v2.0*  
*© 2025 FootballIQ Engineering Team — Confidential*
