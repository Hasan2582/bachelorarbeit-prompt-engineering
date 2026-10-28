
"""Überprüfung der Ergebnisse des Sensitivitätsversuchs.
Erfassung von technischen Fehlern und nicht extrahierbaren Antworten.
Vergleich der Baseline-Templates mit denen des Hauptversuchs."""

from pathlib import Path
import json
import pandas as pd


#Einlesen der Ergebnisse des Sensitivitätsversuchs
data = json.loads(
    Path("results/sensitivity_results.json").read_text(
        encoding="utf-8"
    )
)
df = pd.DataFrame(data)


#Prüfung der Modellantworten
df["valid"] = df["error"].isna()
df["parsed"] = df["valid"] & df["predicted"].notna()
df["correct"] = df["valid"] & df["is_correct"].eq(True)

groups = ["benchmark", "model", "strategy", "variant"]


#Zusammenfassung der Ergebnisse nach Benchmark, Modell, Strategie und Variante
summary = (
    df.groupby(groups)
    .agg(
        total=("question", "size"),
        valid=("valid", "sum"),
        parsed=("parsed", "sum"),
        correct=("correct", "sum")
    )
    .reset_index()
)


#Berechnung der technischen Fehler und nicht extrahierbaren Antworten
summary["errors"] = summary["total"] - summary["valid"]
summary["unparsed"] = summary["valid"] - summary["parsed"]


#Berechnung der Extrahierbarkeit und Genauigkeit in Prozent
summary["format_percent"] = (
    100 * summary["parsed"] / summary["valid"]
)
summary["accuracy_percent"] = (
    100 * summary["correct"] / summary["valid"]
)
summary = summary.round(2)


#Speicherung der Qualitätskontrolle als CSV-Datei
output = Path("results/sensitivity_analysis")
output.mkdir(parents=True, exist_ok=True)
summary.to_csv(output / "quality_audit.csv", index=False)


#Vergleich der Baseline-Templates mit den Templates des Hauptversuchs
for benchmark in ["mmlu_pro", "gsm8k"]:
    for strategy in ["zero_shot", "few_shot"]:

        main = (
            Path("prompt_templates")
            / benchmark
            / f"{strategy}.txt"
        )

        baseline = (
            Path("prompt_templates")
            / benchmark
            / "sensitivity"
            / strategy
            / "baseline.txt"
        )

        main_text = main.read_text(encoding="utf-8").strip()
        baseline_text = baseline.read_text(encoding="utf-8").strip()

        status = (
            "IDENTISCH"
            if main_text == baseline_text
            else "UNTERSCHIEDLICH"
        )

        print(f"{benchmark} | {strategy}: {status}")


#Ausgabe
print("Fertig")
