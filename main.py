
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Database import init_db
from ASRController import router

# Initialize database
init_db()

# Create app
app = FastAPI(
    title="Whisper ASR Backend",
    description="Arabic TV transcription and Whisper fine-tuning pipeline",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "Whisper ASR Backend",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


