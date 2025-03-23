from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import parselmouth
from pdf.report import create_report
from parselmouth.praat import call
import numpy as np
import librosa
from scipy.stats import kurtosis, skew
import io
import soundfile as sf
from scipy.signal import hilbert
from python_speech_features import mfcc
import uvicorn
import traceback
import os
from typing import Dict, Any,List
import io
import tensorflow as tf
from fastapi import APIRouter, File
from PIL import Image
from keras.preprocessing.image import img_to_array
from classifier.train import Train
import cv2 as cv
import imutils
from tensorflow.keras.models import load_model
 
labels: List[str] = ['Healthy', 'Parkinson']

app = FastAPI()

model = joblib.load('models/catboost_model_20250323_032657.pkl')
preprocessor = joblib.load('models/catboost_preprocessor_20250323_032657.pkl')
model = load_model('models/brain-tumor.h5')

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

def analyze_audio(file_path: str) -> Dict[str, Any]:
    try:
    
        sound = parselmouth.Sound(file_path)

        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0] 
        mean_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

     
        intensity = sound.to_intensity()
        intensity_values = intensity.values
        mean_intensity = np.mean(intensity_values)

        
        formants = sound.to_formant_burg()
        midpoint = sound.duration / 2

        
        f1 = formants.get_value_at_time(1, midpoint)
        f2 = formants.get_value_at_time(2, midpoint)
        f3 = formants.get_value_at_time(3, midpoint)

        return {
            'mean_pitch': float(mean_pitch),
            'mean_intensity': float(mean_intensity),
            'f1': float(f1),
            'f2': float(f2),
            'f3': float(f3),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def heuristic_model(mean_pitch: float, mean_intensity: float, f1: float, f2: float, f3: float, thresholds: Dict[str, float]) -> int:
    if (mean_pitch < thresholds['pitch'] and
        mean_intensity < thresholds['intensity'] and
        f1 > thresholds['f1'] and
        f2 > thresholds['f2'] and
        f3 > thresholds['f3']):
        return 1 
    return 0 


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

@app.post("/predict/park")
async def analyze_and_predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        upload_directory = 'temp'
        os.makedirs(upload_directory, exist_ok=True)


        file_path = os.path.join(upload_directory, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        analysis_results = analyze_audio(file_path)
        

        with open('a.txt', 'w') as f:
            f.write(str(analysis_results))

        os.remove(file_path)

        input_data = {
            'mean_pitch': round(analysis_results['mean_pitch'], 2),
            'mean_intensity': round(analysis_results['mean_intensity'], 2),
            'f1': round(analysis_results['f1'], 2),
            'f2': round(analysis_results['f2'], 2),
            'f3': round(analysis_results['f3'], 2)
        }

        thresholds = {
            'pitch': 116.09,
            'intensity': 67.89,
            'f1': 1343.93,
            'f2': 1688.41,
            'f3': 1495.40
        }


        prediction = heuristic_model(
            input_data['mean_pitch'],
            input_data['mean_intensity'],
            input_data['f1'],
            input_data['f2'],
            input_data['f3'],
            thresholds
        )

        result = "Parkinson's" if prediction == 1 else "Not Parkinson's"
        detected= "High" if prediction==1 else "Low"
       
        create_report(detected=detected,pitch=input_data['mean_pitch'],
            intensity=input_data['mean_intensity'],
            f1=input_data['f1'],
            f2=input_data['f2'],
            f3=input_data['f3'],)
        print(detected,input_data['mean_intensity'],input_data['mean_pitch'],input_data['f1'],input_data['f2'],input_data['f3'])
        differences = {
            'pitch_diff': round(input_data['mean_pitch'] - thresholds['pitch'], 2),
            'intensity_diff': round(input_data['mean_intensity'] - thresholds['intensity'], 2),
            'f1_diff': round(input_data['f1'] - thresholds['f1'], 2),
            'f2_diff': round(input_data['f2'] - thresholds['f2'], 2),
            'f3_diff': round(input_data['f3'] - thresholds['f3'], 2)
        }

        response_data = {
            'prediction': result,
            'user_values': input_data,
            'thresholds': thresholds,
            'differences': differences
        }
        

        
        with open('b.txt', 'w') as f:
            f.write(str(response_data))
        headers = {"Content-Disposition": "attachment; filename=voice_analysis_report.pdf"} 
        return FileResponse('voice_analysis_report.pdf',media_type="application/pdf",headers=headers)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict/pneumonia')
def pneumonia_router(image_file: bytes = File(...)):
    model = Train().define_model()
    model.load_weights('models/pneumonia.h5')
    image = Image.open(io.BytesIO(image_file))
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((64, 64))
    image = img_to_array(image)/255.0
    image = image.reshape(1, 64, 64, 1)

 
    prediction = model.predict(image)
    probability = float(prediction[0][0])  
    predicted_class = 'pneumonia' if probability > 0.5 else 'normal'

    return {
        'predicted_class': predicted_class,
        'pneumonia_probability': probability
    }


@app.post("/predict/braintumor", response_model=Dict[str, float])
async def predict_tumor(
    file: UploadFile = File(...)  
):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, detail="File must be an image")
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(400, detail="Could not process image")


        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
        thresh = cv.erode(thresh, None, iterations=2)
        thresh = cv.dilate(thresh, None, iterations=2)

        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if not cnts:
            raise HTTPException(400, detail="No contours found in image")
            
        c = max(cnts, key=cv.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        
        processed_image = cv.resize(new_image, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
        processed_image = processed_image / 255.
        processed_image = processed_image.reshape((1, 240, 240, 3))

        prediction = model.predict(processed_image)
        probability = float(prediction[0][0])

        return {
            "prediction": float(probability > 0.5),  
            "probability": probability
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)