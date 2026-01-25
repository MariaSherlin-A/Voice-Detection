from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np

app = FastAPI()

class AudioRequest(BaseModel):
    audio_base64: str

@app.post("/detect-voice")
def detect_voice(request: AudioRequest):
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        with open("temp.mp3", "wb") as f:
            f.write(audio_bytes)

        y, sr = librosa.load("temp.mp3", sr=None)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        score = float(np.mean(mfcc))

        if score < -200:
            return {
                "classification": "AI_GENERATED",
                "confidence": 0.85,
                "explanation": "Synthetic speech characteristics detected."
            }
        else:
            return {
                "classification": "HUMAN",
                "confidence": 0.80,
                "explanation": "Natural speech patterns detected."
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
