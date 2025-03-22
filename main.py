from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import joblib
import parselmouth
from parselmouth.praat import call
import numpy as np
import librosa
from scipy.stats import kurtosis, skew
import io
import soundfile as sf
from scipy.signal import hilbert
from python_speech_features import mfcc
import uvicorn

app = FastAPI()

model = joblib.load('models/catboost_model_20250323_032657.pkl')
preprocessor = joblib.load('models/catboost_preprocessor_20250323_032657.pkl')

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    prediction_label: str

class VoiceFeatureExtractor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def extract_features_from_audio(self, audio_data, sr):
        sound = parselmouth.Sound(audio_data, sampling_frequency=sr)
        features = {}
        features.update(self._extract_basic_features(sound))
        features.update(self._extract_jitter_shimmer(sound))
        features.update(self._extract_harmonic_features(sound))
        features.update(self._extract_gne_features(sound))
        features.update(self._extract_cepstral_features(audio_data, sr))
        features.update(self._extract_additional_features(sound))
        return features

    def _extract_basic_features(self, sound):
        pitch = sound.to_pitch()
        harmonicity = sound.to_harmonicity()
        intensity = sound.to_intensity()
        
        features = {
            'DPF_a': pitch.selected_array['frequency'].mean(),
            'PFR_a': pitch.selected_array['frequency'].std(),
            'PPE_a': np.ptp(pitch.selected_array['frequency']),
            'PVI_a': np.var(intensity.values),
            'HNR_a': harmonicity.values[harmonicity.values != -200].mean(),
            'DPF_i': pitch.selected_array['frequency'].mean(),
            'PFR_i': pitch.selected_array['frequency'].std(),
            'PPE_i': np.ptp(pitch.selected_array['frequency']),
            'PVI_i': np.var(intensity.values),
            'HNR_i': harmonicity.values[harmonicity.values != -200].mean(),
        }
        return features

    def _extract_jitter_shimmer(self, sound):
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        
        features = {
            'J1_a': call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            'J3_a': call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
            'J5_a': call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            'J55_a': call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
            'S1_a': call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'S3_a': call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'S5_a': call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'S11_a': call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'S55_a': call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'J1_i': call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            'J3_i': call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
            'J5_i': call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            'J55_i': call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
            'S1_i': call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'S3_i': call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'S5_i': call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'S11_i': call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'S55_i': call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        }
        return features

    def _extract_harmonic_features(self, sound):
        harmonicity = sound.to_harmonicity()
        pitch = sound.to_pitch()
        
        harmonics = {}
        for i in range(1, 9):
            harmonics[f'Ha{i}_mu'] = np.mean(harmonicity.values[harmonicity.values != -200])
            harmonics[f'Ha{i}_sd'] = np.std(harmonicity.values[harmonicity.values != -200])
            harmonics[f'Ha{i}_rel'] = harmonics[f'Ha{i}_sd'] / harmonics[f'Ha{i}_mu'] if harmonics[f'Ha{i}_mu'] != 0 else 0
            harmonics[f'Hi{i}_mu'] = np.mean(harmonicity.values[harmonicity.values != -200])
            harmonics[f'Hi{i}_sd'] = np.std(harmonicity.values[harmonicity.values != -200])
            harmonics[f'Hi{i}_rel'] = harmonics[f'Hi{i}_sd'] / harmonics[f'Hi{i}_mu'] if harmonics[f'Hi{i}_mu'] != 0 else 0
        return harmonics

    def _extract_gne_features(self, sound):
        signal = sound.values.T[0]
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        
        features = {
            'GNEa_mu': np.mean(amplitude_envelope),
            'GNEa_sigma': np.std(amplitude_envelope),
            'GNEi_mu': np.mean(amplitude_envelope),
            'GNEi_sigma': np.std(amplitude_envelope)
        }
        return features

    def _extract_cepstral_features(self, audio_data, sr):
        mfcc_features = mfcc(audio_data, samplerate=sr, numcep=12)
        features = {}
        
        for i in range(1, 13):
            features[f'CCa{i}'] = np.mean(mfcc_features[:, i-1])
            features[f'dCCa{i}'] = np.std(mfcc_features[:, i-1])
            features[f'CCi{i}'] = np.mean(mfcc_features[::2, i-1])
            features[f'dCCi{i}'] = np.std(mfcc_features[::2, i-1])
        return features

    def _extract_additional_features(self, sound):
        spectrum = sound.to_spectrum()
        f2 = call(spectrum, "Get centre of gravity", 2)
        
        features = {
            'd_1': np.mean(np.diff(sound.values.T[0])),
            'F2_i': f2,
            'F2_conv': f2 * np.mean(sound.values.T[0])
        }
        return features


column_mapping = {
    'Ha1_mu': 'Ha(1)_mu', 'Ha2_mu': 'Ha(2)_mu', 'Ha3_mu': 'Ha(3)_mu', 'Ha4_mu': 'Ha(4)_mu',
    'Ha5_mu': 'Ha(5)_mu', 'Ha6_mu': 'Ha(6)_mu', 'Ha7_mu': 'Ha(7)_mu', 'Ha8_mu': 'Ha(8)_mu',
    'Ha1_sd': 'Ha(1)_sd', 'Ha2_sd': 'Ha(2)_sd', 'Ha3_sd': 'Ha(3)_sd', 'Ha4_sd': 'Ha(4)_sd',
    'Ha5_sd': 'Ha(5)_sd', 'Ha6_sd': 'Ha(6)_sd', 'Ha7_sd': 'Ha(7)_sd', 'Ha8_sd': 'Ha(8)_sd',
    'Ha1_rel': 'Ha(1)_rel', 'Ha2_rel': 'Ha(2)_rel', 'Ha3_rel': 'Ha(3)_rel', 'Ha4_rel': 'Ha(4)_rel',
    'Ha5_rel': 'Ha(5)_rel', 'Ha6_rel': 'Ha(6)_rel', 'Ha7_rel': 'Ha(7)_rel', 'Ha8_rel': 'Ha(8)_rel',
    'CCa1': 'CCa(1)', 'CCa2': 'CCa(2)', 'CCa3': 'CCa(3)', 'CCa4': 'CCa(4)', 'CCa5': 'CCa(5)',
    'CCa6': 'CCa(6)', 'CCa7': 'CCa(7)', 'CCa8': 'CCa(8)', 'CCa9': 'CCa(9)', 'CCa10': 'CCa(10)',
    'CCa11': 'CCa(11)', 'CCa12': 'CCa(12)', 'dCCa1': 'dCCa(1)', 'dCCa2': 'dCCa(2)', 'dCCa3': 'dCCa(3)',
    'dCCa4': 'dCCa(4)', 'dCCa5': 'dCCa(5)', 'dCCa6': 'dCCa(6)', 'dCCa7': 'dCCa(7)', 'dCCa8': 'dCCa(8)',
    'dCCa9': 'dCCa(9)', 'dCCa10': 'dCCa(10)', 'dCCa11': 'dCCa(11)', 'dCCa12': 'dCCa(12)',
    'Hi1_mu': 'Hi(1)_mu', 'Hi2_mu': 'Hi(2)_mu', 'Hi3_mu': 'Hi(3)_mu', 'Hi4_mu': 'Hi(4)_mu',
    'Hi5_mu': 'Hi(5)_mu', 'Hi6_mu': 'Hi(6)_mu', 'Hi7_mu': 'Hi(7)_mu', 'Hi8_mu': 'Hi(8)_mu',
    'Hi1_sd': 'Hi(1)_sd', 'Hi2_sd': 'Hi(2)_sd', 'Hi3_sd': 'Hi(3)_sd', 'Hi4_sd': 'Hi(4)_sd',
    'Hi5_sd': 'Hi(5)_sd', 'Hi6_sd': 'Hi(6)_sd', 'Hi7_sd': 'Hi(7)_sd', 'Hi8_sd': 'Hi(8)_sd',
    'Hi1_rel': 'Hi(1)_rel', 'Hi2_rel': 'Hi(2)_rel', 'Hi3_rel': 'Hi(3)_rel', 'Hi4_rel': 'Hi(4)_rel',
    'Hi5_rel': 'Hi(5)_rel', 'Hi6_rel': 'Hi(6)_rel', 'Hi7_rel': 'Hi(7)_rel', 'Hi8_rel': 'Hi(8)_rel',
    'CCi1': 'CCi(1)', 'CCi2': 'CCi(2)', 'CCi3': 'CCi(3)', 'CCi4': 'CCi(4)', 'CCi5': 'CCi(5)',
    'CCi6': 'CCi(6)', 'CCi7': 'CCi(7)', 'CCi8': 'CCi(8)', 'CCi9': 'CCi(9)', 'CCi10': 'CCi(10)',
    'CCi11': 'CCi(11)', 'CCi12': 'CCi(12)', 'dCCi1': 'dCCi(1)', 'dCCi2': 'dCCi(2)', 'dCCi3': 'dCCi(3)',
    'dCCi4': 'dCCi(4)', 'dCCi5': 'dCCi(5)', 'dCCi6': 'dCCi(6)', 'dCCi7': 'dCCi(7)', 'dCCi8': 'dCCi(8)',
    'dCCi9': 'dCCi(9)', 'dCCi10': 'dCCi(10)', 'dCCi11': 'dCCi(11)', 'dCCi12': 'dCCi(12)',
    'GNEa_mu': 'GNEa_\\mu', 'GNEa_sigma': 'GNEa_\\sigma', 'GNEi_mu': 'GNEi_\\mu', 'GNEi_sigma': 'GNEi_\\sigma'
}

@app.post("/predict/als", response_model=PredictionOutput)
async def predict_from_audio(file: UploadFile = File(...), age: int = None, sex: str = None):
    try:
        if not file.filename.endswith(('.wav', '.WAV')):
            raise HTTPException(status_code=400, detail="Only WAV files are supported")
        
        if age is None or sex is None:
            raise HTTPException(status_code=400, detail="Age and sex are required")
        
        contents = await file.read()
        audio_data, sr = sf.read(io.BytesIO(contents))
        
        if len(audio_data) / sr < 3:
            raise HTTPException(status_code=400, detail="Audio must be at least 3 seconds long")
        
        extractor = VoiceFeatureExtractor()
        features = extractor.extract_features_from_audio(audio_data, sr)
        
        features['Age'] = age
        features['Sex'] = sex
        
        input_df = pd.DataFrame([features])
        input_df = input_df.rename(columns=column_mapping)
        
        processed_data = preprocessor.transform(input_df)
        probability = model.predict_proba(processed_data)[0][1]
        
        prediction = 1 if probability >= 0.7 else 0
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            prediction_label="ALS Positive" if probability >= 0.7 else "ALS Negative"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)