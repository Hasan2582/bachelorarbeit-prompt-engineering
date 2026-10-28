
"""Durchführung des Hauptversuchs mit MMLU-Pro und GSM8K.
Ausführung der Prompting-Strategien mit beiden Modellen.
Speicherung der Ergebnisse und des Versuchsfortschritts."""

from pathlib import Path

from checkpoint import ensure_manifest, load_jsonl, load_progress, save_final
from run_gsm8k import run_gsm8k
from run_mmlu import run_mmlu


#Verwendete Modelle und Prompting-Strategien
models = ["llama3:8b", "phi3:mini"]
strategies = ["zero_shot", "one_shot", "few_shot", "cot", "cod", "role_prompting"]


#Pfade zu den Datensätzen und Ergebnissen
mmlu_file = Path("data/mmlu_pro_samples50.jsonl")
gsm8k_file = Path("data/gsm8k_samples50.jsonl")
log_path = Path("results/main_progress.jsonl")
result_path = Path("results/eval_run.json")


if __name__ == "__main__":

    #Dateien für die Überprüfung der Versuchskonfiguration
    input_files = [mmlu_file, gsm8k_file] + [
        Path("prompt_templates") / benchmark / f"{strategy}.txt"
        for benchmark in ("mmlu_pro", "gsm8k")
        for strategy in strategies
    ]

    #Überprüfung der Konfiguration und Eingabedateien
    ensure_manifest(
        log_path,
        {"models": models, "strategies": strategies, "temperature": 0.0},
        input_files,
    )

    #Prüfung der gespeicherten Ergebnisse
    if result_path.exists() and not log_path.exists():
        raise FileExistsError("Gespeicherte Aufrufe fehlen")

    #Einlesen der bisherigen Ergebnisse und Datensätze
    rows = load_progress(log_path)
    print(f"Gespeicherte Aufrufe: {len(rows)}")

    mmlu_dataset = load_jsonl(mmlu_file)
    gsm8k_dataset = load_jsonl(gsm8k_file)

    try:
        #Ausführung der beiden Benchmarks
        run_mmlu(models, strategies, mmlu_dataset, rows, log_path)
        run_gsm8k(models, strategies, gsm8k_dataset, rows, log_path)

        #Prüfung auf technische Fehler
        failures = sum(row.get("error") is not None for row in rows.values())

        if failures:
            raise RuntimeError(f"Ollama-Fehler: {failures}")

    finally:
        #Speicherung der Ergebnisse
        save_final(result_path, rows)
        print(f"Ergebnisse gespeichert: {len(rows)}")
