
from pathlib import Path
import pandas as pd

output_dir = Path("results/final_tables")
output_dir.mkdir(parents=True, exist_ok=True)

main = pd.read_csv("results/analysis/summary.csv")
sensitivity = pd.read_csv(
    "results/sensitivity_analysis/sensitivity_summary.csv"
)
quality = pd.read_csv(
    "results/sensitivity_analysis/quality_audit.csv"
)

tables = {
    "main_accuracy": main[[
        "benchmark", "model", "strategy",
        "n_total", "n_valid", "errors",
        "unparsed", "accuracy_percent"
    ]],

    "main_efficiency": main[[
        "benchmark", "model", "strategy",
        "input_tokens", "output_tokens",
        "total_tokens", "latency"
    ]],

    "sensitivity_accuracy": sensitivity[[
        "benchmark", "model", "strategy", "variant",
        "accuracy_percent", "baseline_accuracy_percent",
        "accuracy_change_pp"
    ]],

    "sensitivity_stability": sensitivity[[
        "benchmark", "model", "strategy", "variant",
        "agreement_percent"
    ]],

    "quality_audit": quality
}

for name, table in tables.items():
    table.to_csv(
        output_dir / f"{name}.csv",
        index=False,
        float_format="%.2f"
    )

print("Alle Tabellen wurden unter results/final_tables gespeichert.")