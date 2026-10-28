
"""Durchführung der GSM8K-Aufgaben im Hauptversuch.
Erstellung der Prompts und Abfrage der Modelle über Ollama.
Extraktion und Überprüfung der numerischen Endantworten."""

from pathlib import Path
import re
from decimal import Decimal, InvalidOperation

from checkpoint import append_result, completed_keys, record_key
from ollama_client import query_ollama


NUMBER_PATTERN = r"-?[\d,]+(?:\.\d+)?"


def load_prompt(prompt_name):
    """Einlesen des GSM8K-Prompt-Templates.

    Args:
        prompt_name (str): Name der Prompting-Strategie

    Returns:
        str: Inhalt des Prompt-Templates
    """
    path = Path("prompt_templates") / "gsm8k" / f"{prompt_name}.txt"
    return path.read_text(encoding="utf-8")


def prompt_format(template, item):
    """Einfügen der Aufgabe in das Prompt-Template.

    Args:
        template (str): Inhalt des Prompt-Templates
        item (dict): Aufgabe aus dem GSM8K-Datensatz

    Returns:
        str: Vollständiger Prompt
    """
    return template.format(question=item["question"])


def extract_answer(response_text):
    """Extraktion der Endantwort aus der Modellantwort.

    Args:
        response_text (str): Antwort des Sprachmodells

    Returns:
        str | None: Extrahierte Zahl oder None
    """
    if not response_text:
        return None

    #Suche nach der numerischen Endantwort
    match = re.search(
        r"(?im)^\s*Answer:\s*[$€£]?\s*(-?\d[\d,]*(?:\.\d+)?)",
        response_text
    )

    if match:
        return match.group(1).replace(",", "")

    return None


def extract_ground_truth(answer_text):
    """Extraktion der richtigen Antwort aus der GSM8K-Lösung.

    Args:
        answer_text (str): Referenzlösung aus dem Datensatz

    Returns:
        str | None: Richtige Antwort oder None
    """
    match = re.search(rf"####\s*({NUMBER_PATTERN})", answer_text or "")
    return match.group(1).replace(",", "") if match else None


def answers_equal(predicted, solution):
    """Vergleich der Modellantwort mit der richtigen Antwort.

    Args:
        predicted (str): Extrahierte Modellantwort
        solution (str): Richtige Antwort aus dem Datensatz

    Returns:
        bool: True bei gleicher Antwort, sonst False
    """
    if predicted is None or solution is None:
        return False

    try:
        return Decimal(predicted) == Decimal(solution)
    except InvalidOperation:
        return False


def run_gsm8k(models, strategies, dataset, rows, log_path):
    """Durchführung der GSM8K-Versuche und Speicherung der Ergebnisse.

    Args:
        models (list): Verwendete Sprachmodelle
        strategies (list): Verwendete Prompting-Strategien
        dataset (list): Aufgaben aus dem GSM8K-Datensatz
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
                key = ("gsm8k", model, strategy, None, item_index)

                #Bereits abgeschlossene Aufgaben überspringen
                if key in done:
                    continue

                #Prompt erstellen und Modell abfragen
                prompt = prompt_format(template, item)
                result = query_ollama(model, prompt)

                #Endantwort und Referenzlösung auslesen
                predicted = extract_answer(result.get("response", ""))
                solution = extract_ground_truth(item["answer"])

                if solution is None:
                    raise ValueError(f"Referenzantwort fehlt: {item_index}")

                #Ergebnis und Messwerte speichern
                row = {
                    "benchmark": "gsm8k",
                    "model": model,
                    "strategy": strategy,
                    "item_index": item_index,
                    "question": item["question"],
                    "ground_truth": solution,
                    "predicted": predicted,
                    "is_correct": result.get("error") is None and answers_equal(predicted, solution),
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

            #Ergebnisse der aktuellen Strategie ausgeben
            subset = [
                r for r in rows.values()
                if r["benchmark"] == "gsm8k"
                and r["model"] == model
                and r["strategy"] == strategy
            ]

            correct = sum(r["is_correct"] for r in subset)
            print(f"GSM8K | {model} | {strategy}: {correct}/{len(dataset)} richtig")
