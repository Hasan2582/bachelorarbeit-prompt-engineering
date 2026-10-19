from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


RESULT_FILE = Path("results") / "sensitivity_results.json"
OUTPUT_DIR = Path("results") / "sensitivity_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with open(RESULT_FILE, "r", encoding="utf-8") as file:
    results = json.load(file)


df = pd.DataFrame(results)

df["total_tokens"] = (
    df["input_tokens"]
    + df["output_tokens"]
)


variant_order = [
    "baseline",
    "character",
    "word",
    "sentence",
    "semantic"
]


summary = (
    df.groupby(
        [
            "benchmark",
            "model",
            "strategy",
            "variant"
        ]
    )
    .agg(
        accuracy=("is_correct", "mean"),
        input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"),
        total_tokens=("total_tokens", "mean"),
        latency=("latency", "mean"),
        errors=("error", lambda x: x.notna().sum())
    )
    .reset_index()
)


summary["accuracy_percent"] = (
    summary["accuracy"] * 100
)


baseline_accuracy = (
    summary[
        summary["variant"] == "baseline"
    ][
        [
            "benchmark",
            "model",
            "strategy",
            "accuracy_percent"
        ]
    ]
    .rename(
        columns={
            "accuracy_percent":
            "baseline_accuracy_percent"
        }
    )
)


summary = summary.merge(
    baseline_accuracy,
    on=[
        "benchmark",
        "model",
        "strategy"
    ],
    how="left"
)


summary["accuracy_change_pp"] = (
    summary["accuracy_percent"]
    - summary["baseline_accuracy_percent"]
)


baseline_predictions = (
    df[df["variant"] == "baseline"][
        [
            "benchmark",
            "model",
            "strategy",
            "question",
            "predicted"
        ]
    ]
    .rename(
        columns={
            "predicted": "baseline_prediction"
        }
    )
)


comparison = df.merge(
    baseline_predictions,
    on=[
        "benchmark",
        "model",
        "strategy",
        "question"
    ],
    how="left"
)


comparison["same_as_baseline"] = (
    comparison["predicted"].eq(comparison["baseline_prediction"])
    |
    (
        comparison["predicted"].isna()
        & comparison["baseline_prediction"].isna()
    )
)


stability = (
    comparison.groupby(
        [
            "benchmark",
            "model",
            "strategy",
            "variant"
        ]
    )
    .agg(
        agreement_with_baseline=(
            "same_as_baseline",
            "mean"
        )
    )
    .reset_index()
)


stability["agreement_percent"] = (
    stability["agreement_with_baseline"]
    * 100
)


summary = summary.merge(
    stability,
    on=[
        "benchmark",
        "model",
        "strategy",
        "variant"
    ],
    how="left"
)


summary.to_csv(
    OUTPUT_DIR / "sensitivity_summary.csv",
    index=False
)


for benchmark in summary["benchmark"].unique():

    for strategy in summary["strategy"].unique():

        data = summary[
            (summary["benchmark"] == benchmark)
            & (summary["strategy"] == strategy)
        ].copy()

        data["variant"] = pd.Categorical(
            data["variant"],
            categories=variant_order,
            ordered=True
        )

        data = data.sort_values("variant")


        accuracy = data.pivot(
            index="variant",
            columns="model",
            values="accuracy_percent"
        )

        accuracy.plot(
            kind="bar",
            figsize=(10, 6)
        )

        plt.title(
            f"Prompt-Sensitivität – {benchmark} – {strategy}"
        )

        plt.xlabel("Prompt-Variante")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=0)
        plt.legend(title="Modell")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / f"{benchmark}_{strategy}_accuracy.png",
            dpi=300
        )

        plt.close()


        agreement = data.pivot(
            index="variant",
            columns="model",
            values="agreement_percent"
        )

        agreement.plot(
            kind="bar",
            figsize=(10, 6)
        )

        plt.title(
            f"Übereinstimmung mit Baseline – "
            f"{benchmark} – {strategy}"
        )

        plt.xlabel("Prompt-Variante")
        plt.ylabel("Übereinstimmung (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=0)
        plt.legend(title="Modell")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / f"{benchmark}_{strategy}_agreement.png",
            dpi=300
        )

        plt.close()


print(summary)

print(
    "\nSensitivity-Auswertung wurde unter "
    "'results/sensitivity_analysis/' gespeichert."
)