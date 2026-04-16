
from flask import Flask, request, render_template, redirect
import numpy as np
import pandas as pd
import os

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

application = Flask(__name__)

app = application

def train_if_needed():
    if not os.path.exists("artifacts/model.pkl"):
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        dt = DataTransformation()
        train_arr, test_arr, _ = dt.initiate_data_transformation(train_data, test_data)
        mt = ModelTrainer()
        mt.initiate_model_trainer(train_arr, test_arr)

train_if_needed()

## Route for home page

@app.route('/')
def index():
    return redirect('/predictdata')

@app.route('/predictdata', methods= ['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)