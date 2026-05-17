import os

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
QUEUE_INFERENCE = os.environ.get("QUEUE_INFERENCE", "footballiq.inference")
QUEUE_STATS = os.environ.get("QUEUE_STATS", "footballiq.stats")
QUEUE_REPORTS = os.environ.get("QUEUE_REPORTS", "footballiq.reports")
