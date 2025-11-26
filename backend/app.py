from fastapi import FastAPI
import math
from fastapi.middleware.cors import CORSMiddleware
from models import InsightRequest, InsightResponse
from services.insights import generate_insights   
from fastapi import Query
from services.data_loader import hr_basic_metrics, credit_basic_metrics, get_hr_page, get_cc_page
from fastapi import Body
from services.infer import chat_once
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from services.data_loader import (
    hr_basic_metrics, credit_basic_metrics,
    get_hr_page, get_cc_page
)

app = FastAPI(title="LLM Risk & Performance (Local Models)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _sanitize(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

@app.get("/api/employee/metrics")
def employee_metrics():
    return JSONResponse(content=jsonable_encoder(hr_basic_metrics()))

@app.get("/api/credit/metrics")
def credit_metrics():
    return JSONResponse(content=jsonable_encoder(credit_basic_metrics()))

@app.post("/api/insights", response_model=InsightResponse)
def insights(req: InsightRequest):
    out = generate_insights(req.questions, req.controls)   
    return InsightResponse(
        summary=out.get("summary",""),
        risks=out.get("risks",[]),
        actions=out.get("actions",[]),
        caveats=out.get("caveats",[]),
    )

@app.post("/api/chat")
def chat(payload: dict = Body(...)):
    msg = (payload or {}).get("message", "").strip()
    use_data = bool((payload or {}).get("use_data", True))
    extra = (payload or {}).get("extra_context", None)
    if not msg:
        return {"reply": "Please send a non-empty message."}
    return {"reply": chat_once(msg, use_data=use_data, extra_context=extra)}

@app.get("/api/employee/data")
def employee_data(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
):
    page = get_hr_page(offset=offset, limit=limit)
    return JSONResponse(content=_sanitize(page))

@app.get("/api/credit/data")
def credit_data(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
):
    page = get_cc_page(offset=offset, limit=limit)
    return JSONResponse(content=_sanitize(page))