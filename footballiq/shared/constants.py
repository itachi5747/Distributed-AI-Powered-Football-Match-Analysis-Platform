QUEUE_INFERENCE = "footballiq.inference"
QUEUE_STATS = "footballiq.stats"
QUEUE_REPORTS = "footballiq.reports"
QUEUE_HIGHLIGHTS = "footballiq.highlights"

REDIS_MATCH_STATS = "match:{match_id}:stats"
REDIS_MATCH_PROGRESS = "match:{match_id}:progress"
REDIS_MATCH_FRAMES = "match:{match_id}:frames:latest"
REDIS_SESSION_STATE = "session:{session_id}:state"
REDIS_PUBSUB_MATCH = "pubsub:match:{match_id}"
REDIS_RATELIMIT_USER = "ratelimit:user:{user_id}"
REDIS_JWT_BLACKLIST = "jwt:blacklist:{jti}"

POSSESSION_THRESHOLD_M = 2.0
SPRINT_THRESHOLD_KMH = 25.0
HIGH_INTENSITY_KMH = 19.8
WALK_THRESHOLD_KMH = 7.0

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32
GOAL_CENTRE_M = (105.0, 34.0)

HEATMAP_BINS_X = 52
HEATMAP_BINS_Y = 32
