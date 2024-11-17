from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

model_path = os.path.join(os.path.dirname(__file__), "./appendictis_model.pkl")

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

class InputData(BaseModel):
    PC1: float
    PC2: float
    PC3: float
    PC4: float
    PC5: float
    PC6: float
    PC7: float
    PC8: float
    PC9: float
    PC10: float
    PC11: float
    PC12: float
    PC13: float
    PC14: float
    PC15: float
    PC16: float
    PC17: float
    PC18: float
    PC19: float
    PC20: float
    PC21: float
    PC22: float
    PC23: float
    PC24: float
    PC25: float
    PC26: float
    PC27: float
    PC28: float
    PC29: float
    PC30: float

@app.post("/predict")
async def predict(data: InputData):
    input_data = [list(data.dict().values())]
    print("Expected Features:", model.n_features_in_)

    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}