# THADAM starter project

This folder contains the first Python API implementation for the THADAM maritime freight-planning site. The Bolt frontend ZIP supplied to the conversation was empty, so the frontend is not included or connected yet.

## Backend capabilities

- `GET /health`: readiness check
- `GET /options`: ports and cargo types for frontend selectors
- `POST /forecast`: validate an origin, destination, cargo and quantity; screen compatible vessels against port constraints; return four monthly rate estimates and arrival-window flags. For port pairs without direct rate history, the trained model uses the nearest available route history for the same vessel class and marks the result as `similar_route_model`.
- `POST /optimal-voyage`: select the lowest forecast freight-cost vessel/month for a submitted voyage request
- `GET /optimal-voyage`: reproduce the notebook's fixed Australia-to-Paradip next-order scenario
- `python train.py`: train four direct XGBoost forecast pipelines and write local artifacts

The service uses the provided CSVs in `backend/data`. It excludes observations later than the current month when preparing model data, because the supplied synthetic dataset includes months that may be in the future. The API reports the latest included month and the carry-forward market-input assumption with each response.

## Run locally on macOS

Open Terminal and move into this `backend` folder. Then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` to see the API interface and `http://127.0.0.1:8000/options` to see selector data. The frontend will later call `http://127.0.0.1:8000/forecast` during local development.

If the browser frontend runs on another local port, set `FRONTEND_ORIGIN` to that exact origin before starting Uvicorn. For example: `FRONTEND_ORIGIN=http://localhost:5173`.

## Connect the Bolt frontend later

The frontend should send a JSON `POST` request with these fields to `/forecast` for the four-month vessel forecast and `/optimal-voyage` for the lowest-cost eligible option:

```json
{
  "origin_port": "Newcastle",
  "destination_port": "Paradip",
  "cargo_type": "coal",
  "quantity_mt": 60000,
  "arrival_by": "2026-12-31"
}
```

The port field accepts either a port name or its `port_id`. Quantity is in metric tonnes. Point both frontend requests to the API URL. Keep secrets out of frontend code.

## Deployment outline

The root `render.yaml` prepares a Render web service and is preconfigured to allow the published THADAM frontend at `https://maritime-freight-pla-tpg6.bolt.host`. Push this folder into a private GitHub repository, then create a Render Blueprint from that repository. The service trains the models during build and loads them on startup. The frontend still needs its submit action connected to the eventual Render API URL; the Bolt ZIP export supplied so far was empty, so that source integration is not done yet.

## Limitations to address before public launch

- Forecasts are estimates from synthetic data, not market quotes. Label them clearly.
- The current API checks vessel dimensions, capacity and cargo compatibility, then estimates sailing time from the selected origin. It does not yet reproduce every later notebook step for repositioning costs and cargo-order matching.
- Arrival dates are not guaranteed. A production version should include vessel availability, repositioning, port operations, and user-selected arrival-window semantics.
- Add request throttling and stronger monitoring before advertising the API for substantial public traffic.
