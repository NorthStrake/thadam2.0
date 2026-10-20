from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .modeling import DATA_DIR, ARTIFACT_DIR, GROUP_COLS, TARGET, load_data, load_models, prepare_monthly, forecast_for_series

app = FastAPI(title="THADAM Freight Planning API", version="0.1.0")
origins = [o.strip() for o in os.getenv("FRONTEND_ORIGIN", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])

FRAMES = None
MONTHLY = None
MODELS = None
METADATA = None
FEATURES = None
BASE_DATE = None


class VoyageRequest(BaseModel):
    origin_port: str = Field(min_length=2, max_length=100)
    destination_port: str = Field(min_length=2, max_length=100)
    cargo_type: str = Field(min_length=2, max_length=60)
    quantity_mt: float = Field(gt=0, le=500000)
    arrival_by: date


@app.on_event("startup")
def initialize():
    global FRAMES, MONTHLY, MODELS, METADATA, FEATURES, BASE_DATE
    FRAMES = load_data()
    models_and_metadata = load_models()
    MODELS, METADATA = models_and_metadata
    MONTHLY, FEATURES, _, base = prepare_monthly(FRAMES["freight_rates"], FRAMES["market_indicators"])
    BASE_DATE = base


def find_port(value: str) -> pd.Series:
    ports = FRAMES["ports"]
    matches = ports[(ports.port_id.astype(str).str.casefold() == value.casefold()) |
                    (ports.port_name.astype(str).str.casefold() == value.casefold())]
    if matches.empty:
        raise HTTPException(422, f"Unknown port: {value}")
    return matches.iloc[0]


@app.get("/health")
def health():
    ready = FRAMES is not None and MODELS is not None
    return {"status": "ok" if ready else "starting", "models_loaded": ready}


@app.get("/options")
def options():
    if FRAMES is None:
        raise HTTPException(503, "API is still starting")
    ports = FRAMES["ports"][["port_id", "port_name", "country"]].to_dict(orient="records")
    cargo_types = sorted({v.strip() for cell in FRAMES["ports"].supported_cargo_types.dropna().astype(str)
                          for v in cell.replace("|", ",").split(",") if v.strip()})
    return {"ports": ports, "cargo_types": cargo_types, "quantity_unit": "metric tonnes"}


@app.post("/forecast")
def forecast(request: VoyageRequest):
    if FRAMES is None or MODELS is None:
        raise HTTPException(503, "Forecast service is not ready")
    origin = find_port(request.origin_port)
    destination = find_port(request.destination_port)
    cargo = request.cargo_type.strip().casefold()
    supported = {x.strip().casefold() for x in str(destination.supported_cargo_types).replace("|", ",").split(",")}
    if cargo not in supported:
        raise HTTPException(422, f"{destination.port_name} does not list support for {request.cargo_type}")
    if request.arrival_by < date.today():
        raise HTTPException(422, "Arrival date must be today or later")

    distances = FRAMES["route_distances"]
    route = distances[(distances.origin_port_id == origin.port_id) & (distances.destination_port_id == destination.port_id)]
    if route.empty:
        raise HTTPException(422, "No route distance is available for this port pair")
    route_row = route.iloc[0]
    route_id = str(route_row.route_id)

    vessels = FRAMES["vessels"].copy()
    compatibility = vessels.cargo_compatibility.fillna("").astype(str).str.casefold()
    vessels = vessels[compatibility.map(lambda s: cargo in {x.strip() for x in s.replace("|", ",").split(",")})]
    vessels = vessels[(vessels.length_overall_m <= float(destination.max_vessel_loa_m)) &
                      (vessels.beam_m <= float(destination.max_vessel_beam_m)) &
                      (vessels.loaded_draft_m <= float(destination.max_permissible_draft_m)) &
                      (vessels.deadweight_tonnage_dwt <= float(destination.max_berth_dwt)) &
                      (vessels.usable_cargo_capacity_mt >= request.quantity_mt)]
    if vessels.empty:
        raise HTTPException(422, "No vessels meet the destination-port, cargo and capacity constraints")

    # Forecast from the latest complete month available in the bundled history.
    # The response exposes the data cutoff so users can interpret the scenario.
    latest = MONTHLY.date.max().to_period("M").to_timestamp()
    first_forecast = latest + pd.DateOffset(months=1)
    rate_rows = []
    for vessel_type in sorted(vessels.vessel_type.unique()):
        predictions = forecast_for_series(MONTHLY, FEATURES, BASE_DATE, MODELS, route_id,
                                          str(origin.port_id), str(destination.port_id),
                                          str(vessel_type), first_forecast)
        for p in predictions:
            rate_rows.append({"vessel_type": vessel_type, **p})

    rate_lookup = {(x["vessel_type"], x["horizon_months"]): x["freight_rate_usd_per_tonne"] for x in rate_rows}
    result_vessels = []
    for _, vessel in vessels.iterrows():
        transit_days = float(route_row.distance_nm) / (24 * float(vessel.speed_knots))
        estimated_arrival = (pd.Timestamp.today().normalize() + pd.to_timedelta(transit_days, unit="D")).date()
        forecasts = [{"month": (first_forecast + pd.DateOffset(months=h - 1)).strftime("%Y-%m"),
                      "rate_usd_per_tonne": rate_lookup.get((vessel.vessel_type, h))} for h in (1, 2, 3, 4)]
        eligible_for_window = estimated_arrival <= request.arrival_by
        result_vessels.append({"vessel_id": vessel.vessel_id, "vessel_name": vessel.vessel_name,
                               "vessel_type": vessel.vessel_type, "capacity_mt": int(vessel.usable_cargo_capacity_mt),
                               "estimated_transit_days": round(transit_days, 1),
                               "estimated_arrival_date": estimated_arrival.isoformat(),
                               "meets_arrival_window": eligible_for_window, "monthly_forecasts": forecasts})
    return {"request": request.model_dump(mode="json"), "route": {"route_id": route_id, "distance_nm": float(route_row.distance_nm)},
            "data_cutoff_month": latest.strftime("%Y-%m"), "forecast_assumption": "Latest historical market indicators are carried forward; estimates are scenarios, not guarantees.",
            "vessels_considered": len(result_vessels), "vessels": result_vessels}
