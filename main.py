from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import whisper
import os

app = FastAPI()
model = whisper.load_model("base")  # small + fast

class AudioRequest(BaseModel):
    audio_base64: str

@app.post("/detect-voice")
def detect_voice(request: AudioRequest):
    try:
        # -------- Decode audio --------
        audio_bytes = base64.b64decode(request.audio_base64)
        with open("temp.mp3", "wb") as f:
            f.write(audio_bytes)
        
        # -------- Whisper language detection --------
        result = model.transcribe("temp.mp3", task="transcribe")
        lang_code = result["language"]
        
        language_map = {
            "en": "English",
            "ta": "Tamil",
            "te": "Telugu",
            "hi": "Hindi",
            "ml": "Malayalam"
        }
        detected_language = language_map.get(lang_code, "Unknown")
        
        # -------- IMPROVED AI/Human Classification --------
        y, sr = librosa.load("temp.mp3", sr=None)
        
        # Extract multiple features
        # 1. MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.var(mfcc)  # AI voices often have lower variance
        
        # 2. Zero Crossing Rate (naturalness of speech)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # 3. Spectral Centroid (brightness/naturalness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_var = np.var(spectral_centroid)
        
        # 4. Spectral Rolloff (frequency distribution)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sr_mean = np.mean(spectral_rolloff)
        
        # 5. Pitch variation (human voices have more natural pitch variation)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        
        # 6. Jitter (micro-variations in pitch - humans have natural jitter)
        if len(pitch_values) > 1:
            jitter = np.mean(np.abs(np.diff(pitch_values))) / np.mean(pitch_values) if np.mean(pitch_values) > 0 else 0
        else:
            jitter = 0
        
        # -------- Scoring System --------
        ai_score = 0
        
        # MFCC variance check (AI voices typically < 100)
        if mfcc_var < 100:
            ai_score += 2
        
        # Zero crossing rate (AI voices often have very consistent ZCR)
        if zcr_mean < 0.05 or zcr_mean > 0.15:
            ai_score += 1
        
        # Spectral centroid variance (AI < 1e10)
        if sc_var < 1e10:
            ai_score += 2
        
        # Pitch variation (AI voices have less variation)
        if pitch_std < 50:
            ai_score += 2
        
        # Jitter check (AI voices have very low jitter, < 0.01)
        if jitter < 0.01:
            ai_score += 2
        
        # Energy envelope (AI voices have smoother energy)
        rms = librosa.feature.rms(y=y)
        rms_var = np.var(rms)
        if rms_var < 0.001:
            ai_score += 1
        
        # -------- Classification Decision --------
        # If score >= 5 out of 10, likely AI
        classification = "AI_GENERATED" if ai_score >= 3 else "HUMAN"
        
        return {
            "language": detected_language,
            "language_code": lang_code,
            "classification": classification,
            "confidence_score": ai_score,
            "details": {
                "mfcc_variance": float(mfcc_var),
                "zero_crossing_rate": float(zcr_mean),
                "spectral_centroid_var": float(sc_var),
                "pitch_std": float(pitch_std),
                "jitter": float(jitter),
                "rms_variance": float(rms_var)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        if os.path.exists("temp.mp3"):
            os.remove("temp.mp3")
