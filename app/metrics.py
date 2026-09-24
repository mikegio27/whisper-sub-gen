"""Prometheus counters exposed at /metrics."""

from prometheus_client import Counter, Gauge, Histogram

PROCESSED = Counter("subgen_files_processed_total", "Files successfully subtitled")
FAILED = Counter("subgen_files_failed_total", "Files that failed transcription")
QUEUED = Counter("subgen_files_queued_total", "Files added to the work queue")
SCANS = Counter("subgen_scans_total", "Library scans completed")
MEDIA_SECONDS = Counter("subgen_media_seconds_total", "Seconds of media transcribed")
QUEUE_LENGTH = Gauge("subgen_queue_length", "Files currently waiting in the queue")
PROCESSING = Gauge("subgen_processing", "1 while a file is being transcribed")
CURRENT_PROGRESS = Gauge(
    "subgen_current_file_progress_percent",
    "Progress through the file currently being transcribed",
)
ALIGN_SEGMENTS = Counter(
    "subgen_align_segments_total",
    "Whisper segments by forced-alignment outcome",
    ["result"],  # aligned | fallback
)
ALIGN_ERRORS = Counter(
    "subgen_align_errors_total", "Files where forced alignment failed entirely (whisper times kept)"
)
HALLUCINATIONS = Counter(
    "subgen_hallucinations_dropped_total", "Non-speech hallucinated segments dropped"
)
QA_VIOLATIONS = Histogram(
    "subgen_qa_violations_per_100_cues",
    "Subtitle-standard violations per 100 cues in each written file (qa.score)",
    buckets=(1, 2, 5, 10, 20, 40, 80, 160),
)
RECYCLES = Counter(
    "subgen_memory_recycles_total",
    "Clean restarts between jobs because RSS passed RECYCLE_MEMORY_FRACTION of the limit",
)
