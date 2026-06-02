# 1. Instalamos las dependencias de grado institucional (Polars, PyTorch, FastAPI, Ngrok)
!pip install fastapi uvicorn pyngrok torch polars nest-asyncio -q

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio
from pyngrok import ngrok
import torch
import polars as pl

# 2. Autenticación de Ngrok (Reemplaza 'TU_TOKEN_AQUI' con el Authtoken que sacaste de la página)
ngrok.set_auth_token("TU_TOKEN_AQUI")

# 3. Inicializamos la API
app = FastAPI(title="ALRO SUPREME V4.1 CORE")

# 4. Verificación de Hardware (Para asegurar que la T4 está activa)
device = "CUDA (Tesla T4) ACTIVA" if torch.cuda.is_available() else "CPU Lenta"

@app.get("/")
def home():
    return {
        "status": "ONLINE",
        "motor": "ALRO SUPREME V4.1",
        "hardware": device,
        "mensaje": "Listo para recibir datos de Salesforce"
    }

# (Aquí iría todo el código matemático de ALRO_TeslaT4_Engine que te di en los pasos anteriores)

# 5. Creamos el túnel público
ngrok_tunnel = ngrok.connect(8000)
print("="*60)
print(f"🔥 URL PÚBLICA PARA BOLT Y SALESFORCE: {ngrok_tunnel.public_url} 🔥")
print("="*60)

# 6. Mantenemos el servidor corriendo dentro de Colab
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)