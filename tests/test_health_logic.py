import pandas as pd

from health_logic import (
    build_maintenance_queue,
    classify_health,
    prepare_dataset,
    validate_dataset,
)


def sample_frame(**overrides):
    row = {
        "timestamp": "2026-01-01 10:00:00",
        "machine_id": "MTR-TEST",
        "motor_type": "Induction",
        "location": "Test Bay",
        "rpm": 1450,
        "voltage_v": 400,
        "current_a": 10,
        "power_kw": 6,
        "torque_nm": 50,
        "temperature_c": 65,
        "vibration_mm_s": 2,
        "noise_db": 74,
        "efficiency_pct": 92,
        "anomaly_type": "None",
        "severity": "None",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_healthy_readings_score_healthy():
    prepared = prepare_dataset(sample_frame())
    assert prepared.loc[0, "health_index"] >= 80
    assert prepared.loc[0, "health_status"] == "Healthy"


def test_severe_readings_score_critical():
    prepared = prepare_dataset(
        sample_frame(
            temperature_c=99,
            vibration_mm_s=10,
            current_a=21,
            efficiency_pct=65,
            anomaly_type="Bearing Wear",
            severity="Critical",
        )
    )
    assert prepared.loc[0, "health_index"] < 55
    assert prepared.loc[0, "health_status"] == "Critical"


def test_classification_boundaries():
    assert classify_health(80) == "Healthy"
    assert classify_health(79.9) == "Warning"
    assert classify_health(55) == "Warning"
    assert classify_health(54.9) == "Critical"


def test_validation_reports_missing_columns():
    result = validate_dataset(pd.DataFrame({"timestamp": ["2026-01-01"]}))
    assert not result.valid
    assert "Missing required columns" in result.errors[0]


def test_urgent_queue_for_critical_machine():
    frame = pd.concat(
        [
            sample_frame(timestamp="2026-01-01 08:00:00"),
            sample_frame(
                timestamp="2026-01-01 10:00:00",
                temperature_c=98,
                vibration_mm_s=9,
                anomaly_type="Overheating",
                severity="Critical",
            ),
        ],
        ignore_index=True,
    )
    queue = build_maintenance_queue(prepare_dataset(frame))
    assert queue.loc[0, "priority"] == "Urgent"
