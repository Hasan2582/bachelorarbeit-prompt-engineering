
"""Auswertung der Ergebnisse des Hauptversuchs.
Berechnung der Genauigkeit, Tokens und Latenz.
Speicherung der Ergebnisse in CSV-Dateien und Diagrammen."""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal, InvalidOperation
from run_gsm8k import extract_answer as extract_gsm8k


#Pfade zu den Ergebnissen
RESULT_FILE = Path("results") / "eval_run.json"
OUTPUT_DIR = Path("results") / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#Einlesen der Ergebnisse aus der JSON-Datei
with open(RESULT_FILE, "r", encoding="utf-8") as file:
    results = json.load(file)
df = pd.DataFrame(results)


#Erneute Extraktion der Antworten bei GSM8K
mask = df["benchmark"].eq("gsm8k") & df["error"].isna()
df.loc[mask, "predicted"] = (
    df.loc[mask, "raw_response"].apply(extract_gsm8k)
)


def compare_numbers(row):
    """Vergleich der extrahierten Antwort mit der richtigen Antwort.

    Args:
        row (Series): Zeile mit der extrahierten Antwort und der Referenzantwort.

    Returns:
        bool: True bei gleicher Antwort, sonst False.
    """
    try:
        return (
            Decimal(str(row["predicted"]))
            == Decimal(str(row["ground_truth"]))
        )
    except (InvalidOperation, TypeError):
        return False


#Überprüfung der GSM8K-Antworten
df.loc[mask, "is_correct"] = (
    df.loc[mask].apply(compare_numbers, axis=1)
)


#Berechnung der gesamten Tokens aus Eingabe und Ausgabe
df["total_tokens"] = df["input_tokens"] + df["output_tokens"]


#Auswahl der Modellaufrufe ohne technische Fehler
valid_df = df[df["error"].isna()].copy()
groups = ["benchmark", "model", "strategy"]


#Berechnung der Durchschnittswerte für jede Kombination
#Enthält die Genauigkeit, Tokens, Latenz und nicht extrahierbare Antworten
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


#Erfassung aller Modellaufrufe und der technischen Fehler
attempts = (
    df.groupby(groups)
    .agg(
        n_total=("is_correct", "size"),
        errors=("error", lambda x: x.notna().sum())
    )
    .reset_index()
)


#Zusammenführen der Ergebnisse und Berechnung der Genauigkeit in Prozent
summary = attempts.merge(summary, on=groups, how="left")
summary["accuracy_percent"] = summary["accuracy"] * 100


#Speicherung der zusammengefassten Ergebnisse
summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)


#Erstellung der Diagramme für beide Benchmarks
#Pro Benchmark werden Genauigkeit, Tokenverbrauch und Latenz dargestellt.
#Die X-Achse zeigt die Prompting-Strategien und die Balken die beiden Modelle.
#Auf der Y-Achse stehen die jeweiligen Messwerte.
#Speicherung als PNG-Dateien
for benchmark in summary["benchmark"].unique():

    benchmark_data = summary[
        summary["benchmark"] == benchmark
    ]

    #Diagramm für die Genauigkeit in Prozent
    accuracy = benchmark_data.pivot(
        index="strategy",
        columns="model",
        values="accuracy_percent"
    )

    accuracy.plot(kind="bar", figsize=(10, 6))
    plt.title(f"Accuracy nach Prompting-Strategie – {benchmark}")
    plt.xlabel("Prompting-Strategie")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.legend(title="Modell")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{benchmark}_accuracy.png", dpi=300)
    plt.close()

    #Diagramm für den durchschnittlichen Tokenverbrauch
    tokens = benchmark_data.pivot(
        index="strategy",
        columns="model",
        values="total_tokens"
    )

    #Darstellung der durchschnittlichen Gesamttokens je Strategie und Modell
    tokens.plot(kind="bar", figsize=(10, 6))
    plt.title(f"Durchschnittlicher Tokenverbrauch – {benchmark}")
    plt.xlabel("Prompting-Strategie")
    plt.ylabel("Tokens pro Aufgabe")
    plt.xticks(rotation=45)
    plt.legend(title="Modell")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{benchmark}_tokens.png", dpi=300)
    plt.close()

    #Diagramm für die durchschnittliche Latenz
    latency = benchmark_data.pivot(
        index="strategy",
        columns="model",
        values="latency"
    )

    #Darstellung der durchschnittlichen Antwortzeit in Sekunden
    latency.plot(kind="bar", figsize=(10, 6))
    plt.title(f"Durchschnittliche Antwortzeit – {benchmark}")
    plt.xlabel("Prompting-Strategie")
    plt.ylabel("Latenz (Sekunden)")
    plt.xticks(rotation=45)
    plt.legend(title="Modell")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{benchmark}_latency.png", dpi=300)
    plt.close()


#Ausgabe der Ergebnisse
print(summary)
print("Fertig")
