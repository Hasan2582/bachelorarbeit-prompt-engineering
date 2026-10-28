
"""Auswertung der Ergebnisse des Sensitivitätsversuchs.
Berechnung der Genauigkeit und Übereinstimmung mit der Baseline.
Speicherung der Ergebnisse als CSV-Datei und Diagramme."""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


#Pfade zu den Ergebnissen
RESULT_FILE = Path("results") / "sensitivity_results.json"
OUTPUT_DIR = Path("results") / "sensitivity_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#Einlesen der Ergebnisse aus der JSON-Datei
with open(RESULT_FILE, "r", encoding="utf-8") as file:
    results = json.load(file)
df = pd.DataFrame(results)


#Reihenfolge der untersuchten Promptvarianten
variant_order = [
    "baseline",
    "character",
    "word",
    "sentence",
    "semantic"
]

groups = ["benchmark", "model", "strategy", "variant"]


#Auswahl der Modellaufrufe ohne technische Fehler
valid_df = df[df["error"].isna()].copy()


#Berechnung der Genauigkeit für jede Promptvariante
summary = (
    valid_df.groupby(groups)
    .agg(
        accuracy=("is_correct", "mean")
    )
    .reset_index()
)

summary["accuracy_percent"] = summary["accuracy"] * 100


#Anzahl der Modellaufrufe und technischen Fehler
attempts = (
    df.groupby(groups)
    .agg(
        n_total=("is_correct", "size"),
        errors=("error", lambda x: x.notna().sum())
    )
    .reset_index()
)

summary = attempts.merge(summary, on=groups, how="left")


#Genauigkeit der Baseline für jede Kombination
baseline_accuracy = (
    summary[summary["variant"] == "baseline"][
        ["benchmark", "model", "strategy", "accuracy_percent"]
    ]
    .rename(columns={"accuracy_percent": "baseline_accuracy_percent"})
)

summary = summary.merge(
    baseline_accuracy,
    on=["benchmark", "model", "strategy"],
    how="left"
)


#Berechnung der Genauigkeitsänderung gegenüber der Baseline
summary["accuracy_change_pp"] = (
    summary["accuracy_percent"] - summary["baseline_accuracy_percent"]
)


#Extrahierte Antworten der Baseline
baseline_predictions = (
    valid_df[valid_df["variant"] == "baseline"][
        ["benchmark", "model", "strategy", "question", "predicted"]
    ]
    .rename(columns={"predicted": "baseline_prediction"})
)


#Zuordnung der Baseline-Antworten zu denselben Aufgaben
comparison = valid_df.merge(
    baseline_predictions,
    on=["benchmark", "model", "strategy", "question"],
    how="left"
)


#Vergleich der Antworten mit der jeweiligen Baseline
comparison["same_as_baseline"] = (
    comparison["predicted"].eq(comparison["baseline_prediction"])
    |
    (
        comparison["predicted"].isna()
        & comparison["baseline_prediction"].isna()
    )
)


#Berechnung der Antwortübereinstimmung in Prozent
stability = (
    comparison.groupby(groups)
    .agg(
        agreement_with_baseline=("same_as_baseline", "mean")
    )
    .reset_index()
)

stability["agreement_percent"] = (
    stability["agreement_with_baseline"] * 100
)


#Zusammenführen und Speichern der Ergebnisse
summary = summary.merge(stability, on=groups, how="left")
summary.to_csv(OUTPUT_DIR / "sensitivity_summary.csv", index=False)


#Erstellung der Diagramme für beide Benchmarks und Strategien
#Dargestellt werden die Genauigkeit und die Antwortübereinstimmung
#Die X-Achse zeigt die Promptvarianten und die Balken die beiden Modelle
#Die Y-Achse zeigt die jeweiligen Prozentwerte
#Speicherung als PNG-Dateien
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


        #Diagramm für die Genauigkeit der Promptvarianten
        accuracy = data.pivot(
            index="variant",
            columns="model",
            values="accuracy_percent"
        )

        #Darstellung der Genauigkeit in Prozent
        accuracy.plot(kind="bar", figsize=(10, 6))
        plt.title(f"Prompt-Sensitivität – {benchmark} – {strategy}")
        plt.xlabel("Prompt-Variante")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=0)
        plt.legend(title="Modell")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{benchmark}_{strategy}_accuracy.png",
            dpi=300
        )
        plt.close()


        #Diagramm für die Antwortübereinstimmung mit der Baseline
        agreement = data.pivot(
            index="variant",
            columns="model",
            values="agreement_percent"
        )

        #Darstellung der übereinstimmenden Antworten in Prozent
        agreement.plot(kind="bar", figsize=(10, 6))
        plt.title(f"Übereinstimmung mit Baseline – {benchmark} – {strategy}")
        plt.xlabel("Prompt-Variante")
        plt.ylabel("Übereinstimmung (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=0)
        plt.legend(title="Modell")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{benchmark}_{strategy}_agreement.png",
            dpi=300
        )
        plt.close()


#Ausgabe der Ergebnisse
print(summary)
print("Fertig")
