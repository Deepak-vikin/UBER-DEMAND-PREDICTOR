import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import math
import pickle
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load model
model_path = 'model.pkl'
if os.path.exists(model_path):
    model2 = pickle.load(open(model_path, 'rb'))
else:
    print("Warning: model.pkl not found.")
    model2 = None

@app.get('/')
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict')
def predict(
    request: Request,
    price_per_week: int = Form(...),
    population: int = Form(...),
    monthly_income: int = Form(...),
    average_parking_per_month: int = Form(...)
):
    if model2 is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction_text": "Model not loaded. Please ensure model.pkl exists."
        })

    # The order of features must match what the model expects
    input_data = [price_per_week, population, monthly_income, average_parking_per_month]
    final_data = np.array(input_data).reshape(1, -1)
    
    prediction = model2.predict(final_data)
    output = round(prediction[0], 2)
    
    # Ensure prediction is non-negative
    if output < 0:
        output = 0
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction_text": 'Number of weekly riders should be {}'.format(math.floor(output))
    })

if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)