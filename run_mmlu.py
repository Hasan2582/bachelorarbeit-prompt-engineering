
"""Durchführung der MMLU-Pro-Aufgaben im Hauptversuch.
Erstellung der Prompts und Abfrage der Modelle über Ollama.
Extraktion und Überprüfung der Antworten."""

from pathlib import Path
import re

from checkpoint import append_result, completed_keys, record_key
from ollama_client import query_ollama


def load_prompt(prompt_name):
    """Einlesen des MMLU-Pro-Prompt-Templates.

    Args:
        prompt_name (str): Name der Prompting-Strategie

    Returns:
        str: Inhalt des Prompt-Templates
    """
    path = Path("prompt_templates") / "mmlu_pro" / f"{prompt_name}.txt"
    return path.read_text(encoding="utf-8")


def prompt_format(template, item):
    """Einfügen der Aufgabe und Antwortmöglichkeiten in den Prompt.

    Args:
        template (str): Inhalt des Prompt-Templates
        item (dict): Aufgabe aus dem MMLU-Pro-Datensatz

    Returns:
        str: Vollständiger Prompt
    """
    options = "\n".join(
        f"{chr(65 + i)}) {option}"
        for i, option in enumerate(item["options"])
    )

    return template.format(question=item["question"], options=options)


def extract_answer(response_text):
    """Extraktion der Endantwort aus der Modellantwort.

    Args:
        response_text (str): Antwort des Sprachmodells

    Returns:
        str | None: Extrahierter Antwortbuchstabe oder None
    """
    match = re.search(
        r"Answer:\s*([A-J])\b",
        response_text or "",
        re.IGNORECASE
    )

    return match.group(1).upper() if match else None


def run_mmlu(models, strategies, dataset, rows, log_path):
    """Durchführung der MMLU-Pro-Versuche und Speicherung der Ergebnisse.

    Args:
        models (list): Verwendete Sprachmodelle
        strategies (list): Verwendete Prompting-Strategien
        dataset (list): Aufgaben aus dem MMLU-Pro-Datensatz
        rows (dict): Bisher gespeicherte Ergebnisse
        log_path (Path): Pfad zur Checkpoint-Datei

    Returns:
        None: Ergebnisse werden im Checkpoint gespeichert
    """
    done = completed_keys(rows)
    consecutive_errors = 0

    #Durchlaufen der Modelle, Strategien und Aufgaben
    for model in models:
        for strategy in strategies:
            template = load_prompt(strategy)

            for item_index, item in enumerate(dataset):
                key = ("mmlu_pro", model, strategy, None, item_index)

                #Bereits abgeschlossene Aufgaben überspringen
                if key in done:
                    continue

                #Prompt erstellen und Modell abfragen
                prompt = prompt_format(template, item)
                result = query_ollama(model, prompt)

                #Endantwort und Referenzlösung auslesen
                predicted = extract_answer(result.get("response", ""))
                solution = item["answer"]

                #Ergebnis und Messwerte speichern
                row = {
                    "benchmark": "mmlu_pro",
                    "model": model,
                    "strategy": strategy,
                    "item_index": item_index,
                    "question": item["question"],
                    "ground_truth": solution,
                    "predicted": predicted,
                    "is_correct": result.get("error") is None and predicted == solution,
                    "raw_response": result.get("response", ""),
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

            #Ergebnisse der aktuellen Strategie
            subset = [
                r for r in rows.values()
                if r["benchmark"] == "mmlu_pro"
                and r["model"] == model
                and r["strategy"] == strategy
            ]

            correct = sum(r["is_correct"] for r in subset)
            print(f"MMLU-Pro | {model} | {strategy}: {correct}/{len(dataset)} richtig")
