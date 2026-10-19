from pathlib import Path
import re
from decimal import Decimal, InvalidOperation

from checkpoint import append_result, completed_keys, record_key
from ollama_client import query_ollama


NUMBER_PATTERN = r"-?[\d,]+(?:\.\d+)?"


def load_prompt(prompt_name):
    path = Path("prompt_templates") / "gsm8k" / f"{prompt_name}.txt"
    return path.read_text(encoding="utf-8")


def prompt_format(template, item):
    return template.format(question=item["question"])


def extract_answer(response_text):
    if not response_text:
        return None

    match = re.search(
        r"(?im)^\s*Answer:\s*[$€£]?\s*(-?\d[\d,]*(?:\.\d+)?)",
        response_text
    )

    if match:
        return match.group(1).replace(",", "")

    return None


def extract_ground_truth(answer_text):
    match = re.search(rf"####\s*({NUMBER_PATTERN})", answer_text or "")
    return match.group(1).replace(",", "") if match else None


def answers_equal(predicted, solution):
    if predicted is None or solution is None:
        return False
    try:
        return Decimal(predicted) == Decimal(solution)
    except InvalidOperation:
        return False


def run_gsm8k(models, strategies, dataset, rows, log_path):
    done = completed_keys(rows)
    consecutive_errors = 0

    for model in models:
        for strategy in strategies:
            template = load_prompt(strategy)

            for item_index, item in enumerate(dataset):
                key = ("gsm8k", model, strategy, None, item_index)
                if key in done:
                    continue

                prompt = prompt_format(template, item)
                result = query_ollama(model, prompt)
                predicted = extract_answer(result.get("response", ""))
                solution = extract_ground_truth(item["answer"])
                if solution is None:
                    raise ValueError(f"GSM8K-Ground-Truth fehlt bei Aufgabe {item_index}")

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
                append_result(log_path, rows, row)
                if row["error"] is None:
                    done.add(record_key(row))
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    print(f"Ollama-Fehler: {row['error']}")
                    if consecutive_errors >= 3:
                        raise RuntimeError("3 Ollama-Fehler hintereinander. Lauf angehalten; später erneut starten.")

            subset = [
                r for r in rows.values()
                if r["benchmark"] == "gsm8k" and r["model"] == model
                and r["strategy"] == strategy
            ]
            correct = sum(r["is_correct"] for r in subset)
            print(f"GSM8K | {model} | {strategy}: {correct}/{len(dataset)} richtig; {len(subset)} erfasst")
