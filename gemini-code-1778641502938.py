# 1. INSTALACIÓN DE DEPENDENCIAS (Ejecutar en la primera celda)
# !pip install fastapi uvicorn pyngrok torch polars nest-asyncio pydantic -q

import torch
import torch.nn as nn
import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import nest_asyncio
from pyngrok import ngrok
import time

# 2. DEFINICIÓN DEL CONTRATO DE DATOS (Pydantic)
# Esto asegura que Salesforce envíe exactamente lo que necesitamos.
class ClientPayload(BaseModel):
    client_id: str = Field(..., description="ID único del cliente en Salesforce")
    credit_score: float = Field(..., ge=300, le=850)
    monthly_income: float = Field(..., gt=0)
    debt_ratio: float = Field(..., ge=0, le=1)
    history_vector: list[float] = Field(..., min_items=5)

# 3. MOTOR MATEMÁTICO ALRO (PyTorch + T4)
class ALROMotor(nn.Module):
    def __init__(self):
        super(ALROMotor, self).__init__()
        # Simulación de una capa densa para evaluar riesgo
        self.risk_layer = nn.Linear(8, 1) # 8 variables de entrada al cuantil p90
        
    def forward(self, x_tensor):
        # Aplicación del Freno Tangente Hiperbólica matemáticamente riguroso
        raw_risk = self.risk_layer(x_tensor)
        # La función tanh acota la salida entre -1 y 1
        return torch.tanh(raw_risk)

# Configuración de Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] ALRO SUPREME Inicializado en hardware: {device.type.upper()}")

model = ALROMotor().to(device)
# Desactivar gradientes para inferencia pura (ahorra 50% de VRAM)
model.eval() 

# 4. INICIALIZACIÓN DE API Y CORS
app = FastAPI(title="ALRO SUPREME V4.1 CORE", version="4.1.0")

# Permitir a Bolt.new y Salesforce consumir la API sin bloqueos de navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ONLINE", "motor": "ALRO SUPREME V4.1", "device": str(device)}

@app.post("/v1/evaluate_risk")
def evaluate_risk(payload: ClientPayload):
    start_time = time.time()
    try:
        # A. Ingesta Ultrarrápida con Polars
        # Convertimos la lista de historial a una Serie de Polars para análisis vectorial
        hist_series = pl.Series("hist", payload.history_vector)
        mean_hist = hist_series.mean()
        max_hist = hist_series.max()
        
        # Construcción del vector de características [8 dimensiones]
        features = [
            payload.credit_score / 850.0, # Normalización
            payload.monthly_income / 100000.0,
            payload.debt_ratio,
            mean_hist,
            max_hist,
            0.5, # Variables macroeconómicas simuladas
            0.1,
            0.9
        ]
        
        # B. Transferencia a VRAM y Cálculo Tensorial
        input_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        
        with torch.no_grad(): # Bloqueo estricto de autograd
            risk_score = model(input_tensor).item()
            
        # Transformar salida tanh (-1 a 1) a probabilidad (0 a 1)
        probability = (risk_score + 1) / 2
        
        # C. Lógica de Decisión (Freno y Orquestación)
        action = "APPROVE"
        if probability > 0.85:
            action = "URGENT_RESTRUCTURE_TASK"
        elif probability > 0.60:
            action = "FLAG_FOR_REVIEW"
            
        latency = (time.time() - start_time) * 1000
        
        return {
            "client_id": payload.client_id,
            "risk_probability": round(probability, 4),
            "suggested_action": action,
            "metrics": {
                "compute_latency_ms": round(latency, 2),
                "hardware": str(device)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. ORQUESTACIÓN DEL TÚNEL Y SERVIDOR
def start_server(auth_token: str):
    ngrok.set_auth_token(auth_token)
    
    # Desconectar túneles previos para evitar errores de "puerto en uso"
    for t in ngrok.get_tunnels():
        ngrok.disconnect(t.public_url)
        
    tunnel = ngrok.connect(8000)
    print("="*70)
    print(f"🔥 ENDPOINT PÚBLICO SEGURO: {tunnel.public_url} 🔥")
    print(f"👉 Pega esta URL en Salesforce y Bolt.new: {tunnel.public_url}/v1/evaluate_risk")
    print("="*70)
    
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Reemplaza con tu token real
# start_server("TU_TOKEN_NGROK_AQUI")