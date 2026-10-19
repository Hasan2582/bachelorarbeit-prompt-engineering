from pathlib import Path
import re

from checkpoint import append_result, completed_keys, record_key
from ollama_client import query_ollama


def load_prompt(prompt_name):
    path = Path("prompt_templates") / "mmlu_pro" / f"{prompt_name}.txt"
    return path.read_text(encoding="utf-8")


def prompt_format(template, item):
    options = "\n".join(
        f"{chr(65 + i)}) {option}"
        for i, option in enumerate(item["options"])
    )
    return template.format(question=item["question"], options=options)


def extract_answer(response_text):
    match = re.search(r"Answer:\s*([A-J])\b", response_text or "", re.IGNORECASE)
    return match.group(1).upper() if match else None


def run_mmlu(models, strategies, dataset, rows, log_path):
    done = completed_keys(rows)
    consecutive_errors = 0

    for model in models:
        for strategy in strategies:
            template = load_prompt(strategy)

            for item_index, item in enumerate(dataset):
                key = ("mmlu_pro", model, strategy, None, item_index)
                if key in done:
                    continue

                prompt = prompt_format(template, item)
                result = query_ollama(model, prompt)
                predicted = extract_answer(result.get("response", ""))
                solution = item["answer"]

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
                if r["benchmark"] == "mmlu_pro" and r["model"] == model
                and r["strategy"] == strategy
            ]
            correct = sum(r["is_correct"] for r in subset)
            print(f"MMLU-Pro | {model} | {strategy}: {correct}/{len(dataset)} richtig; {len(subset)} erfasst")
