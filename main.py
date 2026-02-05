import os
import re
import base64
import librosa
import numpy as np
import whisper
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

app = FastAPI()
model = None

def get_whisper_model():
    global model
    if model is None:
        model = whisper.load_model("tiny")
    return model

# ---------------- API KEY ----------------
API_KEY = "mysecretkey123"
API_KEY_NAME = "x-api-key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# ---------------- REQUEST ----------------
class AudioRequest(BaseModel):
    audio_base64: str

# ---------------- API ----------------
@app.post("/detect-voice")
def detect_voice(request: AudioRequest, api_key: str = Depends(verify_api_key)):
    try:
        # -------- BASE64 FIX (MANDATORY) --------
        audio_b64 = request.audio_base64.strip()

        if "," in audio_b64:
            audio_b64 = audio_b64.split(",")[1]

        audio_b64 = re.sub(r"\s+", "", audio_b64)

        padding = len(audio_b64) % 4
        if padding:
            audio_b64 += "=" * (4 - padding)

        audio_bytes = base64.b64decode(audio_b64)

        with open("temp.mp3", "wb") as f:
            f.write(audio_bytes)

        # -------- LANGUAGE DETECTION --------
        model = get_whisper_model()
        result = model.transcribe("temp.mp3", task="detect-language")
        lang_code = result["language"]

        language_map = {
            "en": "English",
            "ta": "Tamil",
            "te": "Telugu",
            "hi": "Hindi",
            "ml": "Malayalam"
        }
        detected_language = language_map.get(lang_code, "Unknown")

        # -------- AUDIO NORMALIZATION --------
        y, sr = librosa.load("temp.mp3", sr=16000, mono=True)

        # -------- FEATURES --------
        mfcc_var = np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
        sc_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))

        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[pitches > 0]
        pitch_std = np.std(pitch_vals) if len(pitch_vals) else 0

        jitter = (
            np.mean(np.abs(np.diff(pitch_vals))) / np.mean(pitch_vals)
            if len(pitch_vals) > 1 else 0
        )

        rms_var = np.var(librosa.feature.rms(y=y))

        # -------- AI HEURISTIC SCORE --------
        ai_score = 0
        if mfcc_var < 100: ai_score += 2
        if zcr_mean < 0.05 or zcr_mean > 0.15: ai_score += 1
        if sc_var < 1e10: ai_score += 2
        if pitch_std < 50: ai_score += 2
        if jitter < 0.01: ai_score += 2
        if rms_var < 0.001: ai_score += 1

        classification = "AI_GENERATED" if ai_score >= 5 else "HUMAN"

        # -------- CONFIDENCE (NORMALIZED) --------
        confidence = (
            min(0.7 + ai_score * 0.04, 0.95)
            if classification == "AI_GENERATED"
            else min(0.7 + (10 - ai_score) * 0.03, 0.95)
        )


        explanation = (
            "Low pitch variation, reduced jitter, and stable spectral features "
            "indicate synthetic voice patterns."
            if classification == "AI_GENERATED"
            else
            "Natural pitch variation, jitter, and energy fluctuations "
            "indicate human speech characteristics."
        )

        return {
            "language": detected_language,
            "language_code": lang_code,
            "classification": classification,
            "confidence_score": round(confidence, 2),
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists("temp.mp3"):
            os.remove("temp.mp3")
