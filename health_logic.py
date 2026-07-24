"""Engineering calculations for the Smart Motor Health Dashboard.

The functions in this module are intentionally rule-based and transparent.
They are suitable for an educational project using simulated data; they are
not a substitute for limits supplied by a real motor manufacturer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SENSOR_LIMITS = {
    "temperature_c": {
        "label": "Temperature",
        "unit": "°C",
        "normal": (0.0, 75.0),
        "warning": (75.0, 85.0),
        "critical": 85.0,
        "direction": "high",
        "weight": 0.28,
    },
    "vibration_mm_s": {
        "label": "Vibration",
        "unit": "mm/s",
        "normal": (0.0, 4.5),
        "warning": (4.5, 7.1),
        "critical": 7.1,
        "direction": "high",
        "weight": 0.28,
    },
    "current_a": {
        "label": "Current",
        "unit": "A",
        "normal": (0.0, 15.0),
        "warning": (15.0, 18.0),
        "critical": 18.0,
        "direction": "high",
        "weight": 0.16,
    },
    "noise_db": {
        "label": "Noise",
        "unit": "dB",
        "normal": (0.0, 85.0),
        "warning": (85.0, 95.0),
        "critical": 95.0,
        "direction": "high",
        "weight": 0.12,
    },
    "efficiency_pct": {
        "label": "Efficiency",
        "unit": "%",
        "normal": (85.0, 100.0),
        "warning": (75.0, 85.0),
        "critical": 75.0,
        "direction": "low",
        "weight": 0.16,
    },
}

REQUIRED_COLUMNS = {
    "timestamp",
    "machine_id",
    "motor_type",
    "location",
    "rpm",
    "voltage_v",
    "current_a",
    "power_kw",
    "torque_nm",
    "temperature_c",
    "vibration_mm_s",
    "noise_db",
    "efficiency_pct",
    "anomaly_type",
    "severity",
}

NUMERIC_COLUMNS = {
    "rpm",
    "voltage_v",
    "current_a",
    "power_kw",
    "torque_nm",
    "temperature_c",
    "vibration_mm_s",
    "noise_db",
    "efficiency_pct",
}

ANOMALY_ACTIONS = {
    "Bearing Wear": "Inspect bearing condition, lubrication, and shaft play.",
    "Overheating": "Check cooling airflow, loading, and winding temperature.",
    "Misalignment": "Verify shaft alignment and coupling condition.",
    "Electrical Overload": "Inspect load, supply balance, and current draw.",
    "Loose Mounting": "Inspect base bolts, mounts, and structural vibration.",
    "None": "Continue routine inspection and trend monitoring.",
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


def validate_dataset(frame: pd.DataFrame) -> ValidationResult:
    """Validate schema and values without modifying the supplied DataFrame."""
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return ValidationResult(False, errors, warnings)

    parsed_dates = pd.to_datetime(frame["timestamp"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_dates:
        errors.append(f"{invalid_dates} timestamp value(s) could not be read.")

    for column in sorted(NUMERIC_COLUMNS):
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = int(converted.isna().sum())
        if invalid:
            errors.append(f"{column} contains {invalid} invalid numeric value(s).")

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate row(s) were found.")
    if frame.empty:
        errors.append("The dataset contains no rows.")
    return ValidationResult(not errors, errors, warnings)


def prepare_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize datatypes and add transparent derived engineering metrics."""
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["machine_id"] = data["machine_id"].astype(str)
    data["anomaly_type"] = data["anomaly_type"].fillna("None").astype(str)
    data["severity"] = (
        data["severity"].fillna("None").astype(str).str.title()
    )
    data["health_index"] = calculate_health_index(data)
    data["risk_score"] = calculate_risk_score(data)
    data["health_status"] = classify_health(data["health_index"])
    data["energy_kwh"] = data["power_kw"] * _sample_hours(data)
    return data.sort_values(["timestamp", "machine_id"]).reset_index(drop=True)


def _sample_hours(data: pd.DataFrame) -> pd.Series:
    """Estimate sample duration by machine, using a safe two-hour fallback."""
    differences = (
        data.sort_values(["machine_id", "timestamp"])
        .groupby("machine_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    typical = differences[(differences > 0) & (differences <= 24)].median()
    fallback = float(typical) if pd.notna(typical) else 2.0
    return differences.where((differences > 0) & (differences <= 24), fallback).fillna(
        fallback
    )


def _sensor_penalty(values: pd.Series, config: dict) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    normal_edge = (
        config["normal"][1]
        if config["direction"] == "high"
        else config["normal"][0]
    )
    critical_edge = float(config["critical"])
    span = max(abs(critical_edge - normal_edge), 1e-6)
    if config["direction"] == "high":
        deviation = (values - normal_edge).clip(lower=0) / span
    else:
        deviation = (normal_edge - values).clip(lower=0) / span
    return (deviation.clip(upper=2.0) * 50.0) * config["weight"]


def calculate_health_index(data: pd.DataFrame) -> pd.Series:
    """Return a 0–100 health score using documented sensor penalties."""
    penalty = pd.Series(0.0, index=data.index)
    for sensor, config in SENSOR_LIMITS.items():
        penalty += _sensor_penalty(data[sensor], config)

    severity_penalty = (
        data["severity"]
        .fillna("None")
        .astype(str)
        .str.title()
        .map({"None": 0, "Low": 4, "Medium": 9, "High": 16, "Critical": 24})
        .fillna(0)
    )
    anomaly_penalty = (
        data["anomaly_type"].fillna("None").astype(str).ne("None").astype(int) * 3
    )
    return (100.0 - penalty - severity_penalty - anomaly_penalty).clip(0, 100).round(1)


def calculate_risk_score(data: pd.DataFrame) -> pd.Series:
    """Return a 0–100 risk score complementary to health and severity."""
    health = calculate_health_index(data)
    severity_risk = (
        data["severity"]
        .fillna("None")
        .astype(str)
        .str.title()
        .map({"None": 0, "Low": 5, "Medium": 12, "High": 22, "Critical": 32})
        .fillna(0)
    )
    return ((100 - health) * 0.78 + severity_risk).clip(0, 100).round(1)


def classify_health(health_index: pd.Series | float) -> pd.Series | str:
    """Classify health: Healthy >= 80, Warning >= 55, otherwise Critical."""
    if np.isscalar(health_index):
        score = float(health_index)
        return "Healthy" if score >= 80 else "Warning" if score >= 55 else "Critical"
    return pd.Series(
        np.select(
            [health_index >= 80, health_index >= 55],
            ["Healthy", "Warning"],
            default="Critical",
        ),
        index=health_index.index,
    )


def latest_by_machine(data: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent row for each machine."""
    return (
        data.sort_values("timestamp")
        .groupby("machine_id", as_index=False)
        .tail(1)
        .sort_values("machine_id")
        .reset_index(drop=True)
    )


def sensor_condition(sensor: str, value: float) -> str:
    config = SENSOR_LIMITS[sensor]
    if config["direction"] == "high":
        if value > config["critical"]:
            return "Critical"
        if value > config["normal"][1]:
            return "Warning"
    else:
        if value < config["critical"]:
            return "Critical"
        if value < config["normal"][0]:
            return "Warning"
    return "Healthy"


def trend_direction(machine_data: pd.DataFrame, points: int = 40) -> str:
    """Classify recent health direction using a simple linear slope."""
    recent = machine_data.sort_values("timestamp").tail(points)
    if len(recent) < 4:
        return "Stable"
    slope = float(np.polyfit(np.arange(len(recent)), recent["health_index"], 1)[0])
    if slope < -0.12:
        return "Deteriorating"
    if slope > 0.12:
        return "Improving"
    return "Stable"


def simulated_rul_hours(machine_data: pd.DataFrame) -> int:
    """Educational RUL proxy based on recent health and deterioration only."""
    recent = machine_data.sort_values("timestamp").tail(50)
    latest = float(recent["health_index"].iloc[-1])
    if len(recent) < 4:
        return int(max(24, latest * 5))
    slope = float(np.polyfit(np.arange(len(recent)), recent["health_index"], 1)[0])
    if slope < -0.05:
        samples_to_critical = max((latest - 45) / abs(slope), 1)
        sample_hours = float(_sample_hours(recent).median())
        return int(np.clip(samples_to_critical * sample_hours, 12, 1200))
    return int(np.clip(latest * 8, 120, 1200))


def build_maintenance_queue(data: pd.DataFrame) -> pd.DataFrame:
    """Build one explainable maintenance recommendation per machine."""
    rows: list[dict] = []
    for machine_id, machine_data in data.groupby("machine_id"):
        machine_data = machine_data.sort_values("timestamp")
        latest = machine_data.iloc[-1]
        recent = machine_data.tail(40)
        trend = trend_direction(machine_data)
        anomalies = recent[recent["anomaly_type"].ne("None")]
        critical_count = int(
            anomalies["severity"].isin(["High", "Critical"]).sum()
        )
        warning_sensors = []
        critical_sensors = []
        for sensor, config in SENSOR_LIMITS.items():
            condition = sensor_condition(sensor, float(latest[sensor]))
            detail = (
                f"{config['label']} {latest[sensor]:.1f} {config['unit']}"
            )
            if condition == "Critical":
                critical_sensors.append(detail)
            elif condition == "Warning":
                warning_sensors.append(detail)

        if (
            latest["health_status"] == "Critical"
            or critical_sensors
            or critical_count >= 2
        ):
            priority, window = "Urgent", "Inspect within 24 hours"
        elif latest["health_status"] == "Warning" or trend == "Deteriorating":
            priority, window = "Schedule Soon", "Inspect within 7 days"
        elif warning_sensors or not anomalies.empty:
            priority, window = "Monitor", "Review within 14 days"
        else:
            priority, window = "Routine", "Next planned service"

        main_anomaly = (
            anomalies["anomaly_type"].mode().iloc[0]
            if not anomalies.empty
            else "None"
        )
        reasons = critical_sensors + warning_sensors
        if main_anomaly != "None":
            reasons.append(f"Repeated event: {main_anomaly}")
        if trend == "Deteriorating":
            reasons.append("Recent Health Index is deteriorating")
        if not reasons:
            reasons.append("Readings remain within educational safe limits")

        rows.append(
            {
                "machine_id": machine_id,
                "priority": priority,
                "health_index": float(latest["health_index"]),
                "risk_score": float(latest["risk_score"]),
                "trend": trend,
                "detected": "; ".join(reasons),
                "recommended_action": ANOMALY_ACTIONS.get(
                    main_anomaly, "Inspect the affected subsystem and verify readings."
                ),
                "inspection_window": window,
                "simulated_rul_hours": simulated_rul_hours(machine_data),
            }
        )

    queue = pd.DataFrame(rows)
    order = {"Urgent": 0, "Schedule Soon": 1, "Monitor": 2, "Routine": 3}
    queue["_order"] = queue["priority"].map(order)
    return (
        queue.sort_values(["_order", "risk_score"], ascending=[True, False])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def normalized_sensor_profile(row: pd.Series) -> tuple[list[str], list[float]]:
    """Return radar labels and a 0–100 'safe condition' score."""
    labels: list[str] = []
    scores: list[float] = []
    for sensor, config in SENSOR_LIMITS.items():
        labels.append(config["label"])
        penalty = float(_sensor_penalty(pd.Series([row[sensor]]), config).iloc[0])
        unweighted = penalty / max(config["weight"], 1e-6)
        scores.append(round(float(np.clip(100 - unweighted, 0, 100)), 1))
    return labels, scores
