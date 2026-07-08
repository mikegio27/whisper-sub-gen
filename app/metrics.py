"""Prometheus counters exposed at /metrics."""

from prometheus_client import Counter, Gauge

PROCESSED = Counter("subgen_files_processed_total", "Files successfully subtitled")
FAILED = Counter("subgen_files_failed_total", "Files that failed transcription")
QUEUED = Counter("subgen_files_queued_total", "Files added to the work queue")
SCANS = Counter("subgen_scans_total", "Library scans completed")
MEDIA_SECONDS = Counter(
    "subgen_media_seconds_total", "Seconds of media transcribed"
)
QUEUE_LENGTH = Gauge("subgen_queue_length", "Files currently waiting in the queue")
PROCESSING = Gauge("subgen_processing", "1 while a file is being transcribed")
CURRENT_PROGRESS = Gauge(
    "subgen_current_file_progress_percent",
    "Progress through the file currently being transcribed",
)
