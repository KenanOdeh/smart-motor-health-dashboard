"""Generate the reproducible 2,100-row simulated motor sensor dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path(__file__).resolve().parents[1] / "data" / "motor_sensor_data.csv"
RNG = np.random.default_rng(42)

MACHINES = [
    ("MTR-01", "Induction", "Assembly Line A", 1475, 10.8, 68, 2.0),
    ("MTR-02", "Induction", "Assembly Line A", 1490, 11.5, 71, 2.5),
    ("MTR-03", "Servo", "Robotics Cell", 1820, 8.2, 64, 1.6),
    ("MTR-04", "Servo", "Robotics Cell", 1760, 9.1, 69, 2.1),
    ("MTR-05", "Synchronous", "Utility Room", 1200, 13.2, 73, 3.1),
    ("MTR-06", "Induction", "Packaging Line", 1510, 12.0, 72, 2.7),
]


def anomaly_for(machine: str, index: int) -> tuple[str, str, float]:
    """Return anomaly type, severity, and a 0–1 fault intensity."""
    if machine == "MTR-04" and index > 260:
        intensity = min((index - 260) / 80, 1.0)
        return "Bearing Wear", "High" if index > 320 else "Medium", intensity
    if machine == "MTR-05" and 120 <= index <= 165:
        return "Overheating", "High", 0.85
    if machine == "MTR-02" and index in range(70, 330, 47):
        return "Misalignment", "Medium", 0.65
    if machine == "MTR-06" and index in range(180, 350, 31):
        return "Electrical Overload", "High", 0.8
    if machine == "MTR-01" and index in {96, 202, 305}:
        return "Loose Mounting", "Low", 0.45
    if RNG.random() < 0.014:
        return RNG.choice(
            ["Bearing Wear", "Misalignment", "Electrical Overload"]
        ), "Low", 0.35
    return "None", "None", 0.0


def build_dataset() -> pd.DataFrame:
    start = pd.Timestamp("2026-01-05 06:00:00")
    timestamps = pd.date_range(start, periods=350, freq="144min")
    records = []
    for machine, motor_type, location, base_rpm, base_current, base_temp, base_vib in MACHINES:
        for index, timestamp in enumerate(timestamps):
            anomaly, severity, fault = anomaly_for(machine, index)
            cycle = np.sin(index / 16) * 0.7
            rpm = base_rpm + RNG.normal(0, 28) - fault * 130
            voltage = 400 + RNG.normal(0, 4.5) - fault * 7
            current = base_current + RNG.normal(0, 0.55) + fault * 7.2
            temperature = base_temp + cycle + RNG.normal(0, 1.8) + fault * 22
            vibration = base_vib + RNG.normal(0, 0.25) + fault * 6.2
            noise = 71 + base_vib * 2 + RNG.normal(0, 1.8) + fault * 18
            efficiency = 92.5 - base_vib * 0.6 + RNG.normal(0, 0.7) - fault * 15
            torque = 52 + base_current * 2.2 + RNG.normal(0, 2.5) - fault * 4
            power = np.sqrt(3) * voltage * current * 0.86 / 1000
            records.append(
                {
                    "timestamp": timestamp,
                    "machine_id": machine,
                    "motor_type": motor_type,
                    "location": location,
                    "rpm": round(max(rpm, 0), 1),
                    "voltage_v": round(max(voltage, 0), 1),
                    "current_a": round(max(current, 0), 2),
                    "power_kw": round(max(power, 0), 2),
                    "torque_nm": round(max(torque, 0), 1),
                    "temperature_c": round(max(temperature, 0), 1),
                    "vibration_mm_s": round(max(vibration, 0), 2),
                    "noise_db": round(max(noise, 0), 1),
                    "efficiency_pct": round(float(np.clip(efficiency, 40, 99)), 1),
                    "anomaly_type": anomaly,
                    "severity": severity,
                }
            )
    return pd.DataFrame(records).sort_values(["timestamp", "machine_id"])


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame = build_dataset()
    frame.to_csv(OUTPUT, index=False)
    print(f"Created {len(frame):,} rows at {OUTPUT}")
