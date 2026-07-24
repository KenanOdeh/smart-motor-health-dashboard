# Smart Motor & Sensor Health Dashboard

A complete, beginner-friendly mechatronics portfolio project by **Kenan Maen
Odeh**. The Streamlit dashboard monitors six simulated motors across five weeks
of sensor readings and turns the data into clear condition and maintenance
decisions.

## What is included

- Six useful pages: Overview, Machine Health, Sensor Trends, Anomaly Analysis,
  Maintenance, and Data Explorer
- Compact interface with one clear page choice and short guidance on every page
- Optional filters and CSV upload kept collapsed until they are needed
- Tabbed charts and single-motor action views to avoid oversized, crowded screens
- 2,100 reproducible simulated sensor records
- Dark and light appearance modes
- Dependent filters with a clean **All machines** option
- Health Index, Risk Score, energy estimate, trends, downtime risk, and a
  clearly labelled simulated RUL estimate
- Interactive Plotly charts, gauges, radar comparison, heatmap, and factory view
- CSV upload validation and fallback to the included dataset
- Downloadable condition, anomaly, maintenance, and filtered-data reports
- Automated tests for the core health and maintenance rules

## Fastest way to run it on Windows

1. Install Python from <https://python.org/downloads/>.
2. During installation, tick **Add Python to PATH**.
3. Double-click `START_DASHBOARD.bat`.
4. The revised dashboard opens at `http://localhost:8502`.

Port `8502` is used intentionally so this revised dashboard can run beside an
older version on port `8501` without both browser tabs showing the same app.
4. Wait while the first setup finishes. The dashboard opens in your browser.

## How to use the dashboard

1. Start on **Overview** to see the fleet condition and highest-risk motor.
2. Choose another page from the left sidebar when you need more detail.
3. On **Machine Health**, choose one motor and read the recommended action.
4. On **Sensor Trends**, choose one sensor and a simple motor view.
5. Open **Filters (optional)** only when you want to narrow the results.
6. Open **Advanced options** only when you need to upload another CSV file.

## Manual Windows commands

Open PowerShell inside this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If the terminal is closed later, reopen PowerShell in this folder and run only:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Uploading another CSV

Use **Upload CSV** in the sidebar. The uploaded file must contain these columns:

`timestamp`, `machine_id`, `motor_type`, `location`, `rpm`, `voltage_v`,
`current_a`, `power_kw`, `torque_nm`, `temperature_c`, `vibration_mm_s`,
`noise_db`, `efficiency_pct`, `anomaly_type`, and `severity`.

If the upload is invalid, the app explains the problem and safely returns to
the included dataset.

## Engineering honesty

This project uses artificial data and transparent educational thresholds. The
Health Index and Risk Score are rule-based. The displayed Remaining Useful
Life is explicitly labelled as a **simulated estimate**; it is not a certified
industrial prediction. Real deployment would require manufacturer limits,
calibrated sensors, operating context, and verified failure history.

## Project structure

```text
smart_motor_health_dashboard/
├── app.py
├── health_logic.py
├── START_DASHBOARD.bat
├── requirements.txt
├── README.md
├── data/
│   └── motor_sensor_data.csv
├── scripts/
│   └── generate_data.py
└── tests/
    └── test_health_logic.py
```
