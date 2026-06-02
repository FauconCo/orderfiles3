import torch
import torch.nn as nn
import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import nest_asyncio
from pyngrok import ngrok
import time

# 1. Contrato de Datos (Protección contra Inyecciones de Salesforce)
class ClientPayload(BaseModel):
    client_id: str
    credit_score: float
    market_volatility: float
    history_vector: list[float]

# 2. Red Neuronal con Freno Tanh
class ALROMotor(nn.Module):
    def __init__(self):
        super(ALROMotor, self).__init__()
        self.linear = nn.Linear(3, 1) # 3 variables agregadas
        
    def forward(self, x):
        return torch.tanh(self.linear(x)) # Freno Matemático

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ALROMotor().to(device)
model.eval()

app = FastAPI(title="ALRO SUPREME V4.1 CORE")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite que Bolt.new consuma la API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/evaluate")
def evaluate_risk(payload: ClientPayload):
    start_t = time.time()
    try:
        # Ingesta Polars (SIMD Vectorization)
        hist_series = pl.Series("hist", payload.history_vector)
        avg_hist = hist_series.mean()
        
        # Preparación de Tensores
        features = [payload.credit_score/850.0, payload.market_volatility, avg_hist]
        input_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        
        # Inferencia sin carga en memoria
        with torch.no_grad():
            risk_raw = model(input_tensor).item()
            probability = (risk_raw + 1) / 2 # Normalizar de -1,1 a 0,1
            
        action = "REESTRUCTURAR_MORA" if probability > 0.80 else "MONITOREAR"
        
        return {
            "client_id": payload.client_id,
            "riesgo_p90": round(probability, 4),
            "accion_agente": action,
            "latencia_ms": round((time.time() - start_t)*1000, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Lanzamiento del Túnel y Servidor
if __name__ == "__main__":
    # IMPORTANTE: Reemplaza con tu Authtoken de ngrok.com
    ngrok.set_auth_token("TU_TOKEN_NGROK") 
    public_url = ngrok.connect(8000).public_url
    print(f"\n[!] URL PÚBLICA PARA BOLT Y SALESFORCE: {public_url}/api/v1/evaluate\n")
    
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
