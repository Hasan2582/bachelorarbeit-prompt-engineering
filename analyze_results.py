from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal, InvalidOperation
from run_gsm8k import extract_answer as extract_gsm8k


RESULT_FILE = Path("results") / "eval_run.json"
OUTPUT_DIR = Path("results") / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with open(RESULT_FILE, "r", encoding="utf-8") as file:
    results = json.load(file)

df = pd.DataFrame(results)
mask = df["benchmark"].eq("gsm8k") & df["error"].isna()

df.loc[mask, "predicted"] = (
    df.loc[mask, "raw_response"].apply(extract_gsm8k)
)

def compare_numbers(row):
    try:
        return (
            Decimal(str(row["predicted"]))
            == Decimal(str(row["ground_truth"]))
        )
    except (InvalidOperation, TypeError):
        return False

df.loc[mask, "is_correct"] = (
    df.loc[mask].apply(compare_numbers, axis=1)
)

df.to_csv(
    OUTPUT_DIR / "reparsed_records.csv",
    index=False
)

df["total_tokens"] = (
    df["input_tokens"]
    + df["output_tokens"]
)


valid_df = df[df["error"].isna()].copy()

groups = ["benchmark", "model", "strategy"]

summary = (
    valid_df.groupby(groups)
    .agg(
        n_valid=("is_correct", "size"),
        accuracy=("is_correct", "mean"),
        input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"),
        total_tokens=("total_tokens", "mean"),
        latency=("latency", "mean"),
        unparsed=("predicted", lambda x: x.isna().sum())
    )
    .reset_index()
)

attempts = (
    df.groupby(groups)
    .agg(
        n_total=("is_correct", "size"),
        errors=("error", lambda x: x.notna().sum())
    )
    .reset_index()
)

summary = attempts.merge(summary, on=groups, how="left")

summary["accuracy_percent"] = (
    summary["accuracy"] * 100
)

summary.to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False
)

df[df["error"].notna()].to_csv(
    OUTPUT_DIR / "technical_errors.csv",
    index=False
)


for benchmark in summary["benchmark"].unique():

    benchmark_data = summary[
        summary["benchmark"] == benchmark
    ]

    accuracy = benchmark_data.pivot(
        index="strategy",
        columns="model",
        values="accuracy_percent"
    )

    accuracy.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title(
        f"Accuracy nach Prompting-Strategie – {benchmark}"
    )
    plt.xlabel("Prompting-Strategie")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.legend(title="Modell")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{benchmark}_accuracy.png",
        dpi=300
    )

    plt.close()


    tokens = benchmark_data.pivot(
        index="strategy",
        columns="model",
        values="total_tokens"
    )

    tokens.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title(
        f"Durchschnittlicher Tokenverbrauch – {benchmark}"
    )
    plt.xlabel("Prompting-Strategie")
    plt.ylabel("Tokens pro Aufgabe")
    plt.xticks(rotation=45)
    plt.legend(title="Modell")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{benchmark}_tokens.png",
        dpi=300
    )

    plt.close()


    latency = benchmark_data.pivot(
        index="strategy",
        columns="model",
        values="latency"
    )

    latency.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title(
        f"Durchschnittliche Antwortzeit – {benchmark}"
    )
    plt.xlabel("Prompting-Strategie")
    plt.ylabel("Latenz (Sekunden)")
    plt.xticks(rotation=45)
    plt.legend(title="Modell")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{benchmark}_latency.png",
        dpi=300
    )

    plt.close()


print(summary)

print(
    "\nAuswertung wurde unter "
    "'results/analysis/' gespeichert."
)