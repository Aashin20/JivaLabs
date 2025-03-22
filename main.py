from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()


model = joblib.load('models/catboost_model_20250323_032657.pkl')
preprocessor = joblib.load('models/catboost_preprocessor_20250323_032657.pkl')

class PredictionInput(BaseModel):
    Sex: str
    Age: float
    J1_a: float
    J3_a: float
    J5_a: float
    J55_a: float
    S1_a: float
    S3_a: float
    S5_a: float
    S11_a: float
    S55_a: float
    DPF_a: float
    PFR_a: float
    PPE_a: float
    PVI_a: float
    HNR_a: float
    GNEa_mu: float
    GNEa_sigma: float
    Ha1_mu: float
    Ha2_mu: float
    Ha3_mu: float
    Ha4_mu: float
    Ha5_mu: float
    Ha6_mu: float
    Ha7_mu: float
    Ha8_mu: float
    Ha1_sd: float
    Ha2_sd: float
    Ha3_sd: float
    Ha4_sd: float
    Ha5_sd: float
    Ha6_sd: float
    Ha7_sd: float
    Ha8_sd: float
    Ha1_rel: float
    Ha2_rel: float
    Ha3_rel: float
    Ha4_rel: float
    Ha5_rel: float
    Ha6_rel: float
    Ha7_rel: float
    Ha8_rel: float
    CCa1: float
    CCa2: float
    CCa3: float
    CCa4: float
    CCa5: float
    CCa6: float
    CCa7: float
    CCa8: float
    CCa9: float
    CCa10: float
    CCa11: float
    CCa12: float
    dCCa1: float
    dCCa2: float
    dCCa3: float
    dCCa4: float
    dCCa5: float
    dCCa6: float
    dCCa7: float
    dCCa8: float
    dCCa9: float
    dCCa10: float
    dCCa11: float
    dCCa12: float
    J1_i: float
    J3_i: float
    J5_i: float
    J55_i: float
    S1_i: float
    S3_i: float
    S5_i: float
    S11_i: float
    S55_i: float
    DPF_i: float
    PFR_i: float
    PPE_i: float
    PVI_i: float
    HNR_i: float
    GNEi_mu: float
    GNEi_sigma: float
    Hi1_mu: float
    Hi2_mu: float
    Hi3_mu: float
    Hi4_mu: float
    Hi5_mu: float
    Hi6_mu: float
    Hi7_mu: float
    Hi8_mu: float
    Hi1_sd: float
    Hi2_sd: float
    Hi3_sd: float
    Hi4_sd: float
    Hi5_sd: float
    Hi6_sd: float
    Hi7_sd: float
    Hi8_sd: float
    Hi1_rel: float
    Hi2_rel: float
    Hi3_rel: float
    Hi4_rel: float
    Hi5_rel: float
    Hi6_rel: float
    Hi7_rel: float
    Hi8_rel: float
    CCi1: float
    CCi2: float
    CCi3: float
    CCi4: float
    CCi5: float
    CCi6: float
    CCi7: float
    CCi8: float
    CCi9: float
    CCi10: float
    CCi11: float
    CCi12: float
    dCCi1: float
    dCCi2: float
    dCCi3: float
    dCCi4: float
    dCCi5: float
    dCCi6: float
    dCCi7: float
    dCCi8: float
    dCCi9: float
    dCCi10: float
    dCCi11: float
    dCCi12: float
    d_1: float
    F2_i: float
    F2_conv: float

    class Config:
        json_schema_extra = {
            "example": {
                "Sex": "M",
                "Age": 58,
                "J1_a": 0.321817,
                "J3_a": 0.141230,
                "J5_a": 0.199128,
                "J55_a": 0.923634,
                "S1_a": 6.044559,
                "S3_a": 3.196477,
                "S5_a": 3.770575,
                "S11_a": 5.709480,
                "S55_a": 10.080498,
                "DPF_a": 58.483755,
                "PFR_a": 0.261201,
                "PPE_a": 0.953932,
                "PVI_a": 0.497905,
                "HNR_a": 5.611603,
                "GNEa_mu": 0.916190,
                "GNEa_sigma": 0.033908,
                "Ha1_mu": -30.714386,
                "Ha2_mu": -16.287423,
                "Ha3_mu": -21.036437,
                "Ha4_mu": -10.515995,
                "Ha5_mu": -18.079135,
                "Ha6_mu": -22.050308,
                "Ha7_mu": -27.556129,
                "Ha8_mu": -32.988570,
                "Ha1_sd": 4.137014,
                "Ha2_sd": 5.378604,
                "Ha3_sd": 3.296739,
                "Ha4_sd": 7.365126,
                "Ha5_sd": 5.773458,
                "Ha6_sd": 9.784643,
                "Ha7_sd": 3.718647,
                "Ha8_sd": 7.867399,
                "Ha1_rel": 0.028693,
                "Ha2_rel": 0.046155,
                "Ha3_rel": 0.041096,
                "Ha4_rel": 0.055925,
                "Ha5_rel": 0.041924,
                "Ha6_rel": 0.031412,
                "Ha7_rel": 0.031975,
                "Ha8_rel": 0.024476,
                "CCa1": -7.456684,
                "CCa2": -8.269332,
                "CCa3": -12.385919,
                "CCa4": -11.454670,
                "CCa5": -8.556208,
                "CCa6": 1.708060,
                "CCa7": -0.324135,
                "CCa8": -18.335941,
                "CCa9": 6.885555,
                "CCa10": -8.679375,
                "CCa11": -3.506784,
                "CCa12": 1.763508,
                "dCCa1": -0.002462,
                "dCCa2": 0.013113,
                "dCCa3": 0.008769,
                "dCCa4": -0.012167,
                "dCCa5": -0.018855,
                "dCCa6": 0.015665,
                "dCCa7": -0.013625,
                "dCCa8": -0.001902,
                "dCCa9": 0.010679,
                "dCCa10": 0.009690,
                "dCCa11": 0.001612,
                "dCCa12": -0.012439,
                "J1_i": 0.271146,
                "J3_i": 0.120561,
                "J5_i": 0.161298,
                "J55_i": 0.927794,
                "S1_i": 3.995223,
                "S3_i": 1.992887,
                "S5_i": 2.312774,
                "S11_i": 3.277052,
                "S55_i": 9.752598,
                "DPF_i": 59.165613,
                "PFR_i": 0.367941,
                "PPE_i": 1.003757,
                "PVI_i": 0.548737,
                "HNR_i": 6.831568,
                "GNEi_mu": 0.902368,
                "GNEi_sigma": 0.034524,
                "Hi1_mu": -31.849881,
                "Hi2_mu": -13.100419,
                "Hi3_mu": -18.301512,
                "Hi4_mu": -23.891639,
                "Hi5_mu": -25.370107,
                "Hi6_mu": -48.535125,
                "Hi7_mu": -50.115098,
                "Hi8_mu": -59.795340,
                "Hi1_sd": 1.959521,
                "Hi2_sd": 4.319681,
                "Hi3_sd": 4.419067,
                "Hi4_sd": 4.801380,
                "Hi5_sd": 4.241107,
                "Hi6_sd": 9.526454,
                "Hi7_sd": 5.651439,
                "Hi8_sd": 7.702146,
                "Hi1_rel": 0.029578,
                "Hi2_rel": 0.057405,
                "Hi3_rel": 0.044013,
                "Hi4_rel": 0.034852,
                "Hi5_rel": 0.033771,
                "Hi6_rel": 0.017223,
                "Hi7_rel": 0.017932,
                "Hi8_rel": 0.014815,
                "CCi1": -5.329959,
                "CCi2": 5.290671,
                "CCi3": -19.320494,
                "CCi4": -25.520891,
                "CCi5": 3.141565,
                "CCi6": -7.162548,
                "CCi7": -11.358263,
                "CCi8": -5.392510,
                "CCi9": 2.585950,
                "CCi10": -3.587968,
                "CCi11": 10.674193,
                "CCi12": -5.543137,
                "dCCi1": 0.001830,
                "dCCi2": -0.014360,
                "dCCi3": 0.013787,
                "dCCi4": 0.016449,
                "dCCi5": -0.021777,
                "dCCi6": 0.016809,
                "dCCi7": -0.024467,
                "dCCi8": -0.005300,
                "dCCi9": 0.051874,
                "dCCi10": -0.037710,
                "dCCi11": -0.026549,
                "dCCi12": -0.021149,
                "d_1": 4.825476,
                "F2_i": 2526.285657,
                "F2_conv": 833.498083
            }
        }

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    prediction_label: str

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
       
        input_df = pd.DataFrame([input_data.dict()])
        
       
        column_mapping = {
            'Ha1_mu': 'Ha(1)_mu',
            'Ha2_mu': 'Ha(2)_mu',
            'Ha3_mu': 'Ha(3)_mu',
            'Ha4_mu': 'Ha(4)_mu',
            'Ha5_mu': 'Ha(5)_mu',
            'Ha6_mu': 'Ha(6)_mu',
            'Ha7_mu': 'Ha(7)_mu',
            'Ha8_mu': 'Ha(8)_mu',
            'Ha1_sd': 'Ha(1)_sd',
            'Ha2_sd': 'Ha(2)_sd',
            'Ha3_sd': 'Ha(3)_sd',
            'Ha4_sd': 'Ha(4)_sd',
            'Ha5_sd': 'Ha(5)_sd',
            'Ha6_sd': 'Ha(6)_sd',
            'Ha7_sd': 'Ha(7)_sd',
            'Ha8_sd': 'Ha(8)_sd',
            'Ha1_rel': 'Ha(1)_rel',
            'Ha2_rel': 'Ha(2)_rel',
            'Ha3_rel': 'Ha(3)_rel',
            'Ha4_rel': 'Ha(4)_rel',
            'Ha5_rel': 'Ha(5)_rel',
            'Ha6_rel': 'Ha(6)_rel',
            'Ha7_rel': 'Ha(7)_rel',
            'Ha8_rel': 'Ha(8)_rel',
            'CCa1': 'CCa(1)',
            'CCa2': 'CCa(2)',
            'CCa3': 'CCa(3)',
            'CCa4': 'CCa(4)',
            'CCa5': 'CCa(5)',
            'CCa6': 'CCa(6)',
            'CCa7': 'CCa(7)',
            'CCa8': 'CCa(8)',
            'CCa9': 'CCa(9)',
            'CCa10': 'CCa(10)',
            'CCa11': 'CCa(11)',
            'CCa12': 'CCa(12)',
            'dCCa1': 'dCCa(1)',
            'dCCa2': 'dCCa(2)',
            'dCCa3': 'dCCa(3)',
            'dCCa4': 'dCCa(4)',
            'dCCa5': 'dCCa(5)',
            'dCCa6': 'dCCa(6)',
            'dCCa7': 'dCCa(7)',
            'dCCa8': 'dCCa(8)',
            'dCCa9': 'dCCa(9)',
            'dCCa10': 'dCCa(10)',
            'dCCa11': 'dCCa(11)',
            'dCCa12': 'dCCa(12)',
            'Hi1_mu': 'Hi(1)_mu',
            'Hi2_mu': 'Hi(2)_mu',
            'Hi3_mu': 'Hi(3)_mu',
            'Hi4_mu': 'Hi(4)_mu',
            'Hi5_mu': 'Hi(5)_mu',
            'Hi6_mu': 'Hi(6)_mu',
            'Hi7_mu': 'Hi(7)_mu',
            'Hi8_mu': 'Hi(8)_mu',
            'Hi1_sd': 'Hi(1)_sd',
            'Hi2_sd': 'Hi(2)_sd',
            'Hi3_sd': 'Hi(3)_sd',
            'Hi4_sd': 'Hi(4)_sd',
            'Hi5_sd': 'Hi(5)_sd',
            'Hi6_sd': 'Hi(6)_sd',
            'Hi7_sd': 'Hi(7)_sd',
            'Hi8_sd': 'Hi(8)_sd',
            'Hi1_rel': 'Hi(1)_rel',
            'Hi2_rel': 'Hi(2)_rel',
            'Hi3_rel': 'Hi(3)_rel',
            'Hi4_rel': 'Hi(4)_rel',
            'Hi5_rel': 'Hi(5)_rel',
            'Hi6_rel': 'Hi(6)_rel',
            'Hi7_rel': 'Hi(7)_rel',
            'Hi8_rel': 'Hi(8)_rel',
            'CCi1': 'CCi(1)',
            'CCi2': 'CCi(2)',
            'CCi3': 'CCi(3)',
            'CCi4': 'CCi(4)',
            'CCi5': 'CCi(5)',
            'CCi6': 'CCi(6)',
            'CCi7': 'CCi(7)',
            'CCi8': 'CCi(8)',
            'CCi9': 'CCi(9)',
            'CCi10': 'CCi(10)',
            'CCi11': 'CCi(11)',
            'CCi12': 'CCi(12)',
            'dCCi1': 'dCCi(1)',
            'dCCi2': 'dCCi(2)',
            'dCCi3': 'dCCi(3)',
            'dCCi4': 'dCCi(4)',
            'dCCi5': 'dCCi(5)',
            'dCCi6': 'dCCi(6)',
            'dCCi7': 'dCCi(7)',
            'dCCi8': 'dCCi(8)',
            'dCCi9': 'dCCi(9)',
            'dCCi10': 'dCCi(10)',
            'dCCi11': 'dCCi(11)',
            'dCCi12': 'dCCi(12)',
            'GNEa_mu': 'GNEa_\\mu',
            'GNEa_sigma': 'GNEa_\\sigma',
            'GNEi_mu': 'GNEi_\\mu',
            'GNEi_sigma': 'GNEi_\\sigma'
        }
        
     
        input_df = input_df.rename(columns=column_mapping)
        
      
        processed_data = preprocessor.transform(input_df)
        
      
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            prediction_label="ALS Positive" if prediction == 1 else "ALS Negative"
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
import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", 
                host="0.0.0.0", 
                port=8000, 
                reload=True)