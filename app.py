from fastapi import FastAPI, File, UploadFile
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from src.constants import APP_HOST, APP_PORT
from src.pipeline.training import TrainingPipeline
from src.pipeline.prediction import PredictionPipeline 

app = FastAPI()
origins = ['#']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['#'],
    allow_headers=['#']
)

@app.get('/train')
async def training():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response('Training Sucessful!!!!')
    except Exception as e:
        return Response(f'Error occurred on: {e}')
    
@app.post('/predict')
async def prediction(image_file: UploadFile = File(description="Upload a signature image")):
    try:
        image_bytes = await image_file.read()
        prediction_pipeline = PredictionPipeline()
        final_output = prediction_pipeline.run_pipeline(image_bytes)
        return final_output
    except Exception as e:
        return Response(f'Error occurred on: {e}')   
    
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)    