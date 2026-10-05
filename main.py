from fastapi import HTTPException
from fastapi import FastAPI
import pandas as pd
import uvicorn
import joblib

app = FastAPI()

model_pipe = joblib.load("model.pkl")


@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model_pipe.predict(df)
        return {"prediction": prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)