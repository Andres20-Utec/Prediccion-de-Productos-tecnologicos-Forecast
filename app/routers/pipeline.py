from fastapi import APIRouter
from ..paquete.pipeline_runner import run_pipeline

router = APIRouter()

@router.get("/run_pipeline")
def execute_pipeline():
    """
    Endpoint para ejecutar el pipeline de predicción.
    """
    result = run_pipeline()
    return result
