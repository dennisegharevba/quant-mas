"""
Research ledger - mandatory experiment logging infrastructure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LEDGER_PATH = Path(__file__).resolve().parent / "experiments.csv"

LEDGER_COLUMNS = [
    "experiment_id", "timestamp_utc", "model_name", "script",
    "universe", "n_instruments", "horizons", "test_fraction", "lookback_days",
    "hyperparameters_json", "n_combinations_tested",
    "oos_metrics_json", "costs_included",
    "notes", "decision", "influenced_later_decisions",
]


@dataclass
class Experiment:
    model_name: str
    script: str
    universe: list[str]
    horizons: list[int]
    test_fraction: float
    lookback_days: int
    hyperparameters: dict = field(default_factory=dict)
    n_combinations_tested: int = 0
    oos_metrics: dict = field(default_factory=dict)
    costs_included: bool = False
    notes: str = ""
    decision: str = ""
    influenced_later_decisions: str = ""


def _next_experiment_id() -> int:
    if not LEDGER_PATH.exists():
        return 1
    df = pd.read_csv(LEDGER_PATH)
    if len(df) == 0:
        return 1
    return int(df["experiment_id"].max()) + 1


def log_experiment(exp: Experiment) -> int:
    exp_id = _next_experiment_id()
    row = {
        "experiment_id": exp_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": exp.model_name,
        "script": exp.script,
        "universe": ";".join(exp.universe),
        "n_instruments": len(exp.universe),
        "horizons": ";".join(str(h) for h in exp.horizons),
        "test_fraction": exp.test_fraction,
        "lookback_days": exp.lookback_days,
        "hyperparameters_json": json.dumps(exp.hyperparameters),
        "n_combinations_tested": exp.n_combinations_tested,
        "oos_metrics_json": json.dumps(exp.oos_metrics),
        "costs_included": exp.costs_included,
        "notes": exp.notes,
        "decision": exp.decision,
        "influenced_later_decisions": exp.influenced_later_decisions,
    }
    file_exists = LEDGER_PATH.exists()
    df_row = pd.DataFrame([row], columns=LEDGER_COLUMNS)
    df_row.to_csv(LEDGER_PATH, mode="a", header=not file_exists, index=False)
    return exp_id


def load_ledger() -> pd.DataFrame:
    if not LEDGER_PATH.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.read_csv(LEDGER_PATH)


def print_ledger() -> None:
    df = load_ledger()
    if len(df) == 0:
        print("Ledger is empty - no experiments logged yet.")
        return
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    cols = ["experiment_id", "timestamp_utc", "model_name", "n_instruments",
            "horizons", "n_combinations_tested", "decision"]
    print(df[cols].to_string(index=False))
    print(f"\nTotal experiments logged: {len(df)}")
    print(f"Total combinations tested across all experiments: {df['n_combinations_tested'].sum()}")


if __name__ == "__main__":
    print_ledger()