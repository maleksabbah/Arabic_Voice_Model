from Database import get_db
from TranscriptionService import TranscriptionService

db = next(get_db())
transcriber = TranscriptionService(db, model_name="large-v3")
transcriber.transcribe_series(series_id=1, language="ar")