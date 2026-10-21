from __future__ import annotations

import math
import os
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .modeling import DATA_DIR, ARTIFACT_DIR, GROUP_COLS, TARGET, load_data, load_models, prepare_monthly, forecast_for_series

app = FastAPI(title="THADAM Freight Planning API", version="0.1.0")
origins = [
    "http://localhost:5173",
    "https://maritime-freight-pla-7z45.bolt.host",
    *[o.strip() for o in os.getenv("FRONTEND_ORIGIN", "").split(",") if o.strip()],
]
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


@app.get("/optimal-voyage")
def optimal_voyage():
    """Reproduce the notebook's hardcoded Paradip next-order scenario.

    Assumptions intentionally mirror the notebook: an initial 60,000 nm leg
    from 2026-10-01 to Paradip (PRT_014), follow-on destinations PRT_014–020,
    $1.50/nm repositioning and $5,000 per idle day.
    """
    if FRAMES is None or MODELS is None:
        raise HTTPException(503, "Forecast service is not ready")

    current_port = "PRT_014"
    start_date = pd.Timestamp("2026-10-01")
    initial_distance_nm = 60_000.0
    reposition_cost_per_nm = 1.5
    idle_cost_per_day = 5_000.0
    destinations = [f"PRT_{i:03d}" for i in range(14, 21)]
    months = ["2026-11", "2026-12", "2027-01", "2027-02"]

    ports = FRAMES["ports"].set_index("port_id")
    if current_port not in ports.index:
        raise HTTPException(500, "The hardcoded Paradip port is missing from port data")
    paradip = ports.loc[current_port]
    vessel_pool = FRAMES["vessels"].copy()
    vessel_pool = vessel_pool[
        (vessel_pool.length_overall_m <= float(paradip.max_vessel_loa_m)) &
        (vessel_pool.beam_m <= float(paradip.max_vessel_beam_m)) &
        (vessel_pool.loaded_draft_m <= float(paradip.max_permissible_draft_m)) &
        (vessel_pool.deadweight_tonnage_dwt <= float(paradip.max_berth_dwt))
    ].copy()

    route_distances = FRAMES["route_distances"]
    orders = FRAMES["cargo_orders"].copy()
    orders["order_time"] = pd.to_datetime(orders.order_time, errors="coerce")
    orders["delivery_deadline"] = pd.to_datetime(orders.delivery_deadline, errors="coerce")
    # The Colab scenario resets every valid delivery deadline to 2027 before
    # matching orders (cell 33). Keep this explicit so the web result follows
    # the same scenario instead of rejecting historical-year deadlines.
    valid_deadlines = orders.delivery_deadline.notna()
    orders.loc[valid_deadlines, "delivery_deadline"] = orders.loc[
        valid_deadlines, "delivery_deadline"
    ].map(lambda deadline: (
        deadline.replace(year=2027, day=28)
        if deadline.month == 2 and deadline.day == 29
        else deadline.replace(year=2027)
    ))
    orders["quantity_mt"] = pd.to_numeric(orders.quantity_mt, errors="coerce")
    active_statuses = {"pending", "open", "active", "available"}
    orders = orders[orders.order_status.astype(str).str.casefold().isin(active_statuses)]
    orders = orders.dropna(subset=["order_time", "delivery_deadline", "quantity_mt"])

    latest = MONTHLY.date.max().to_period("M").to_timestamp()
    first_forecast = latest + pd.DateOffset(months=1)
    candidates = []

    for destination_id in destinations:
        route = route_distances[
            (route_distances.origin_port_id.astype(str) == current_port) &
            (route_distances.destination_port_id.astype(str) == destination_id)
        ]
        if route.empty:
            continue
        route_row = route.iloc[0]
        route_id = str(route_row.route_id)
        distance_nm = float(route_row.distance_nm)
        matching_orders = orders[
            (orders.origin_port_id.astype(str) == current_port) &
            (orders.destination_port_id.astype(str) == destination_id)
        ]
        if matching_orders.empty:
            continue

        destination = ports.loc[destination_id] if destination_id in ports.index else None
        if destination is None:
            continue
        compatible_vessels = vessel_pool.copy()
        compatible_vessels = compatible_vessels[
            (compatible_vessels.length_overall_m <= float(destination.max_vessel_loa_m)) &
            (compatible_vessels.beam_m <= float(destination.max_vessel_beam_m)) &
            (compatible_vessels.loaded_draft_m <= float(destination.max_permissible_draft_m)) &
            (compatible_vessels.deadweight_tonnage_dwt <= float(destination.max_berth_dwt))
        ]
        if compatible_vessels.empty:
            continue

        rate_by_type = {}
        for vessel_type in compatible_vessels.vessel_type.astype(str).unique():
            predictions = forecast_for_series(
                MONTHLY, FEATURES, BASE_DATE, MODELS, route_id, current_port,
                destination_id, vessel_type, first_forecast
            )
            rate_by_type[vessel_type] = {
                item["month"]: float(item["freight_rate_usd_per_tonne"])
                for item in predictions
            }

        for _, vessel in compatible_vessels.iterrows():
            speed = float(vessel.speed_knots)
            if speed <= 0:
                continue
            initial_arrival = start_date + pd.to_timedelta(initial_distance_nm / (24 * speed), unit="D")
            next_leg_days = round(distance_nm / (24 * speed))
            first_arrival = (initial_arrival + pd.to_timedelta(next_leg_days, unit="D")).round("D")

            for month_index, month in enumerate(months):
                arrival = first_arrival + pd.DateOffset(months=month_index)
                eligible = matching_orders[
                    (matching_orders.quantity_mt <= float(vessel.deadweight_tonnage_dwt)) &
                    (matching_orders.delivery_deadline >= arrival)
                ].copy()
                if eligible.empty:
                    continue
                eligible["idle_days"] = eligible.order_time.map(
                    lambda available: math.ceil(max(0.0, (available - arrival).total_seconds() / 86400))
                )
                eligible["reposition_cost_usd"] = distance_nm * reposition_cost_per_nm
                eligible["idle_cost_usd"] = eligible.idle_days * idle_cost_per_day
                eligible["additional_cost_usd"] = eligible.reposition_cost_usd + eligible.idle_cost_usd
                order = eligible.sort_values(["additional_cost_usd", "delivery_deadline"]).iloc[0]
                rate = rate_by_type.get(str(vessel.vessel_type), {}).get(month)
                if rate is None:
                    continue
                freight_cost = rate * float(vessel.deadweight_tonnage_dwt)
                candidates.append({
                    "vessel_id": str(vessel.vessel_id),
                    "vessel_name": str(vessel.vessel_name),
                    "vessel_type": str(vessel.vessel_type),
                    "month": month,
                    "selected_order_id": str(order.order_id),
                    "origin_port_id": current_port,
                    "destination_port_id": destination_id,
                    "arrival_date": arrival.date().isoformat(),
                    "order_deadline": order.delivery_deadline.date().isoformat(),
                    "order_quantity_mt": int(order.quantity_mt),
                    "deadweight_tonnage_dwt": int(vessel.deadweight_tonnage_dwt),
                    "distance_nm": distance_nm,
                    "freight_rate_usd_per_tonne": round(rate, 3),
                    "predicted_freight_cost_usd": round(freight_cost, 2),
                    "reposition_cost_usd": round(float(order.reposition_cost_usd), 2),
                    "idle_days": int(order.idle_days),
                    "idle_cost_usd": round(float(order.idle_cost_usd), 2),
                    "additional_cost_usd": round(float(order.additional_cost_usd), 2),
                    "total_cost_usd": round(freight_cost + float(order.additional_cost_usd), 2),
                })

    if not candidates:
        return {
            "status": "no_eligible_match",
            "message": "No open order matched the hardcoded Paradip scenario and vessel constraints.",
            "assumptions": {
                "initial_origin": "Australia (hardcoded in notebook scenario)",
                "initial_destination_port_id": current_port,
                "initial_distance_nm": initial_distance_nm,
                "start_date": start_date.date().isoformat(),
                "reposition_cost_usd_per_nm": reposition_cost_per_nm,
                "idle_cost_usd_per_day": idle_cost_per_day,
            },
        }

    best = min(candidates, key=lambda item: item["total_cost_usd"])
    return {
        "status": "success",
        "optimal": best,
        "candidates_evaluated": len(candidates),
        "data_cutoff_month": latest.strftime("%Y-%m"),
        "assumptions": {
            "initial_origin": "Australia (hardcoded in notebook scenario)",
            "initial_destination_port_id": current_port,
            "initial_distance_nm": initial_distance_nm,
            "start_date": start_date.date().isoformat(),
            "reposition_cost_usd_per_nm": reposition_cost_per_nm,
            "idle_cost_usd_per_day": idle_cost_per_day,
            "freight_cost_basis": "predicted USD/tonne multiplied by vessel deadweight tonnage",
            "order_selection": "lowest repositioning plus idle cost among eligible open orders for each vessel and month",
        },
    }
