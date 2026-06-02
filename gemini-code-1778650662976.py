from fastapi import FastAPI
from ALRO_T4_CUDA_CORE import ALRO_TeslaT4_Engine
import json

app = FastAPI()
motor = ALRO_TeslaT4_Engine()

@app.post("/alro/evaluate")
async def evaluate_risk(payload: dict):
    decision = motor.process_client_payload(payload)
    return json.loads(decision)