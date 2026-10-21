"""Optional next-order matching, isolated so it can be removed independently."""

from __future__ import annotations

import math
from typing import Callable

import pandas as pd

REPOSITION_COST_PER_NM = 1.50
IDLE_COST_PER_DAY = 5_000.0
ACTIVE_ORDER_STATUSES = {"open", "pending", "active", "available"}


def _tokens(value: object) -> set[str]:
    return {part.strip().casefold().replace(" ", "_")
            for part in str(value or "").replace("|", ",").split(",") if part.strip()}


def _fits_port(vessel: pd.Series, port: pd.Series) -> bool:
    return (
        float(vessel.length_overall_m) <= float(port.max_vessel_loa_m)
        and float(vessel.beam_m) <= float(port.max_vessel_beam_m)
        and float(vessel.loaded_draft_m) <= float(port.max_permissible_draft_m)
        and float(vessel.deadweight_tonnage_dwt) <= float(port.max_berth_dwt)
    )


def find_best_next_order(
    *,
    vessel: pd.Series,
    current_port_id: str,
    available_at: pd.Timestamp,
    frames: dict[str, pd.DataFrame],
    forecast_months: list[str],
    route_rate_forecaster: Callable[[str, str, str], dict[str, dict]],
) -> dict | None:
    """Find the lowest all-in-rate eligible order after a vessel becomes free.

    Forecasts are USD/tonne and are multiplied by the matched order's quantity.
    Repositioning is empty travel from the vessel's current port to the order's
    origin. Idle time starts when it reaches that origin and ends when the order
    and the relevant forecast month are available. Orders are ranked by their
    all-in effective rate (freight plus repositioning and idle costs per tonne),
    avoiding a bias toward choosing the smallest order by total dollars alone.
    """
    ports = frames["ports"].copy().set_index("port_id", drop=False)
    routes = frames["route_distances"].copy()
    orders = frames["cargo_orders"].copy()
    orders["order_time"] = pd.to_datetime(orders["order_time"], errors="coerce")
    orders["delivery_deadline"] = pd.to_datetime(orders["delivery_deadline"], errors="coerce")
    orders["quantity_mt"] = pd.to_numeric(orders["quantity_mt"], errors="coerce")
    orders["_status"] = orders["order_status"].astype(str).str.strip().str.casefold()
    orders = orders[
        orders["_status"].isin(ACTIVE_ORDER_STATUSES)
        & orders["order_time"].notna()
        & orders["delivery_deadline"].notna()
        & orders["quantity_mt"].notna()
        & (orders["quantity_mt"] > 0)
    ].copy()

    if current_port_id not in ports.index:
        return None
    current_port = ports.loc[current_port_id]
    if isinstance(current_port, pd.DataFrame):
        current_port = current_port.iloc[0]

    speed = float(vessel.speed_knots)
    capacity = float(vessel.usable_cargo_capacity_mt)
    if speed <= 0 or capacity <= 0:
        return None

    vessel_cargo = _tokens(vessel.cargo_compatibility)
    candidates: list[dict] = []

    for _, order in orders.iterrows():
        origin_id = str(order.origin_port_id)
        destination_id = str(order.destination_port_id)
        if origin_id not in ports.index or destination_id not in ports.index:
            continue
        origin = ports.loc[origin_id]
        destination = ports.loc[destination_id]
        if isinstance(origin, pd.DataFrame):
            origin = origin.iloc[0]
        if isinstance(destination, pd.DataFrame):
            destination = destination.iloc[0]

        cargo = str(order.cargo_type).strip().casefold().replace(" ", "_")
        quantity = float(order.quantity_mt)
        if cargo not in vessel_cargo or quantity > capacity:
            continue
        if cargo not in _tokens(origin.supported_cargo_types) or cargo not in _tokens(destination.supported_cargo_types):
            continue
        if not _fits_port(vessel, origin) or not _fits_port(vessel, destination):
            continue

        if origin_id == current_port_id:
            reposition_nm = 0.0
        else:
            reposition_route = routes[
                (routes.origin_port_id.astype(str) == str(current_port_id))
                & (routes.destination_port_id.astype(str) == origin_id)
            ]
            if reposition_route.empty:
                continue
            reposition_nm = float(reposition_route.iloc[0].distance_nm)

        order_route = routes[
            (routes.origin_port_id.astype(str) == origin_id)
            & (routes.destination_port_id.astype(str) == destination_id)
        ]
        if order_route.empty:
            continue
        order_distance_nm = float(order_route.iloc[0].distance_nm)
        reposition_days = reposition_nm / (24.0 * speed)
        at_pickup = available_at + pd.to_timedelta(reposition_days, unit="D")
        order_time = pd.Timestamp(order.order_time)
        deadline = pd.Timestamp(order.delivery_deadline)

        # The order can be served in any of the model's available forecast months.
        for month in forecast_months:
            month_start = pd.Timestamp(f"{month}-01")
            month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
            load_start = max(at_pickup, order_time, month_start)
            if load_start >= month_end:
                continue
            idle_days = int(math.ceil(max(0.0, (load_start - at_pickup).total_seconds() / 86400.0)))
            load_days = quantity / max(float(origin.cargo_handling_rate_tonnes_per_day), 1.0)
            sailing_days = order_distance_nm / (24.0 * speed)
            delivery_arrival = load_start + pd.to_timedelta(load_days + sailing_days, unit="D")
            if delivery_arrival > deadline:
                continue

            rates = route_rate_forecaster(origin_id, destination_id, month)
            forecast = rates.get(month)
            if not forecast or forecast.get("rate_usd_per_tonne") is None:
                continue
            rate = float(forecast["rate_usd_per_tonne"])
            freight_cost = rate * quantity
            reposition_cost = reposition_nm * REPOSITION_COST_PER_NM
            idle_cost = idle_days * IDLE_COST_PER_DAY
            candidates.append({
                "order_id": str(order.order_id),
                "cargo_type": str(order.cargo_type),
                "quantity_mt": quantity,
                "origin_port_id": origin_id,
                "origin_port_name": str(origin.port_name),
                "destination_port_id": destination_id,
                "destination_port_name": str(destination.port_name),
                "order_available_at": order_time.date().isoformat(),
                "delivery_deadline": deadline.date().isoformat(),
                "forecast_month": month,
                "loading_start": load_start.isoformat(),
                "estimated_delivery_arrival": delivery_arrival.isoformat(),
                "order_route_distance_nm": round(order_distance_nm, 1),
                "freight_rate_usd_per_tonne": round(rate, 3),
                "forecast_basis": forecast.get("forecast_basis"),
                "freight_cost_usd": round(freight_cost, 2),
                "reposition_distance_nm": round(reposition_nm, 1),
                "reposition_transit_days": round(reposition_days, 2),
                "reposition_cost_usd": round(reposition_cost, 2),
                "idle_days": idle_days,
                "idle_cost_usd": round(idle_cost, 2),
                "additional_cost_usd": round(reposition_cost + idle_cost, 2),
                "effective_rate_usd_per_tonne": round(
                    (freight_cost + reposition_cost + idle_cost) / quantity, 3
                ),
                "total_cost_usd": round(freight_cost + reposition_cost + idle_cost, 2),
            })

    if not candidates:
        return None
    return min(candidates, key=lambda item: (
        item["effective_rate_usd_per_tonne"], item["total_cost_usd"],
        item["delivery_deadline"], item["order_id"]
    ))
