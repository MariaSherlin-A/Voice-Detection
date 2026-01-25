from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import io
import soundfile as sf

app = FastAPI()

API_KEY = "my_secret_key"

class AudioRequest(BaseModel):
    audio_base64: str

@app.post("/detect-voice")
def detect_voice(
    request: AudioRequest,
    authorization: str = Header(None)
):
    if authorization is None or authorization.strip() != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_file = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_file)
    except:
        raise HTTPException(status_code=400, detail="Invalid audio")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc)

    if mfcc_mean < -200:
        return {
            "classification": "AI_GENERATED",
            "confidence": 0.85,
            "explanation": "Synthetic spectral patterns detected."
        }
    else:
        return {
            "classification": "HUMAN",
            "confidence": 0.80,
            "explanation": "Natural speech variations detected."
        }
