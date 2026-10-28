
"""Durchführung des Sensitivitätsversuchs mit MMLU-Pro und GSM8K.
Untersuchung der Promptvarianten mit beiden Sprachmodellen.
Speicherung der Ergebnisse und der bisherigen Modellaufrufe."""

from pathlib import Path
import argparse
import json

from checkpoint import (
    append_result, completed_keys, ensure_manifest, load_jsonl,
    load_progress, record_key, save_final, save_json_atomic,
)
from ollama_client import query_ollama
from run_mmlu import prompt_format as format_mmlu, extract_answer as extract_mmlu
from run_gsm8k import (
    prompt_format as format_gsm8k, extract_answer as extract_gsm8k,
    extract_ground_truth, answers_equal,
)


#Verwendete Modelle, Strategien und Promptvarianten
models = ["llama3:8b", "phi3:mini"]
strategies = ["zero_shot", "few_shot"]
variants = ["baseline", "character", "word", "sentence", "semantic"]


def run(benchmark):
    """Durchführung des Sensitivitätsversuchs für einen Benchmark.

    Args:
        benchmark (str): Name des Benchmarks

    Returns:
        None: Ergebnisse werden als JSON-Dateien gespeichert
    """

    #Pfade zu den Datensätzen und Ergebnissen
    data_file = Path("data") / f"{benchmark}_samples50.jsonl"
    log_path = Path("results") / f"sensitivity_{benchmark}_progress.jsonl"
    result_path = Path("results") / f"sensitivity_{benchmark}.json"

    #Dateien für die Prüfung der Versuchskonfiguration
    input_files = [data_file] + [
        Path("prompt_templates") / benchmark / "sensitivity" / strategy / f"{variant}.txt"
        for strategy in strategies
        for variant in variants
    ]

    #Überprüfung der Konfiguration und Eingabedateien
    ensure_manifest(
        log_path,
        {
            "benchmark": benchmark,
            "models": models,
            "strategies": strategies,
            "variants": variants,
            "temperature": 0.0
        },
        input_files,
    )

    #Prüfung der gespeicherten Ergebnisse
    if result_path.exists() and not log_path.exists():
        raise FileExistsError("Gespeicherte Aufrufe fehlen")

    #Einlesen des Datensatzes und der bisherigen Ergebnisse
    dataset = load_jsonl(data_file)
    rows = load_progress(log_path)
    done = completed_keys(rows)
    consecutive_errors = 0

    print(f"Gespeicherte Aufrufe ({benchmark}): {len(rows)}")

    try:
        #Durchlaufen der Modelle, Strategien und Varianten
        for model in models:
            for strategy in strategies:
                for variant in variants:

                    #Einlesen des jeweiligen Prompt-Templates
                    template_path = (
                        Path("prompt_templates") / benchmark / "sensitivity"
                        / strategy / f"{variant}.txt"
                    )
                    template = template_path.read_text(encoding="utf-8")

                    #Durchlaufen der Aufgaben
                    for item_index, item in enumerate(dataset):
                        key = (benchmark, model, strategy, variant, item_index)

                        #Bereits abgeschlossene Aufgaben überspringen
                        if key in done:
                            continue

                        #Prompt erstellen und Referenzantwort bestimmen
                        if benchmark == "mmlu_pro":
                            prompt = format_mmlu(template, item)
                            solution = item["answer"]
                        else:
                            prompt = format_gsm8k(template, item)
                            solution = extract_ground_truth(item["answer"])

                            if solution is None:
                                raise ValueError(f"Referenzantwort fehlt: {item_index}")

                        #Modell abfragen und Antwort extrahieren
                        result = query_ollama(model, prompt)
                        response_text = result.get("response", "")

                        predicted = (
                            extract_mmlu(response_text)
                            if benchmark == "mmlu_pro"
                            else extract_gsm8k(response_text)
                        )

                        #Vergleich mit der richtigen Antwort
                        is_correct = (
                            predicted == solution
                            if benchmark == "mmlu_pro"
                            else answers_equal(predicted, solution)
                        )

                        #Ergebnis und Messwerte speichern
                        row = {
                            "benchmark": benchmark,
                            "model": model,
                            "strategy": strategy,
                            "variant": variant,
                            "item_index": item_index,
                            "question": item["question"],
                            "ground_truth": solution,
                            "predicted": predicted,
                            "is_correct": result.get("error") is None and is_correct,
                            "raw_response": response_text,
                            "input_tokens": result.get("input_tokens", 0),
                            "output_tokens": result.get("output_tokens", 0),
                            "latency": result.get("latency", 0),
                            "error": result.get("error"),
                        }

                        #Fortschritt speichern und Fehler prüfen
                        append_result(log_path, rows, row)

                        if row["error"] is None:
                            done.add(record_key(row))
                            consecutive_errors = 0
                        else:
                            consecutive_errors += 1
                            print(f"Ollama-Fehler: {row['error']}")

                            if consecutive_errors >= 3:
                                raise RuntimeError("3 Ollama-Fehler hintereinander")

                    #Ergebnisse der aktuellen Promptvariante
                    subset = [
                        r for r in rows.values()
                        if r["model"] == model
                        and r["strategy"] == strategy
                        and r["variant"] == variant
                    ]

                    correct = sum(r["is_correct"] for r in subset)
                    print(
                        f"{benchmark} | {model} | {strategy} | {variant}: "
                        f"{correct}/{len(dataset)} richtig"
                    )

        #Prüfung auf technische Fehler
        failures = sum(row.get("error") is not None for row in rows.values())

        if failures:
            raise RuntimeError(f"Ollama-Fehler: {failures}")

    finally:
        #Speicherung der bisherigen Ergebnisse
        save_final(result_path, rows)
        print(f"Ergebnisse gespeichert: {len(rows)}")

    #Zusammenführen der Ergebnisse beider Benchmarks
    first = Path("results/sensitivity_mmlu_pro.json")
    second = Path("results/sensitivity_gsm8k.json")

    if first.exists() and second.exists():
        combined = load_json_file(first) + load_json_file(second)
        save_json_atomic("results/sensitivity_results.json", combined)
        print("Ergebnisse zusammengeführt")


def load_json_file(path):
    """Einlesen einer JSON-Datei.

    Args:
        path (Path): Pfad zur JSON-Datei

    Returns:
        list: Eingelesene Ergebnisse
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


#Start des Sensitivitätsversuchs
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["mmlu_pro", "gsm8k"])
    args = parser.parse_args()

    run(args.benchmark)
