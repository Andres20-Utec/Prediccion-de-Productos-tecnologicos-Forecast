from fastapi import FastAPI
from app.routers import pipeline

app = FastAPI(title="Forecasting API", version="1.0.0")

# Incluir el enrutador
app.include_router(pipeline.router)

# Si deseas una ruta principal para chequear el estado
@app.get("/")
def root():
    return {"message": "API de predicción activa."}
