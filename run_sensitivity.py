from pathlib import Path
import argparse

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


models = ["llama3:8b", "phi3:mini"]
strategies = ["zero_shot", "few_shot"]
variants = ["baseline", "character", "word", "sentence", "semantic"]


def run(benchmark):
    data_file = Path("data") / f"{benchmark}_samples50.jsonl"
    log_path = Path("results") / f"sensitivity_{benchmark}_progress.jsonl"
    result_path = Path("results") / f"sensitivity_{benchmark}.json"

    input_files = [data_file] + [
        Path("prompt_templates") / benchmark / "sensitivity" / strategy / f"{variant}.txt"
        for strategy in strategies for variant in variants
    ]
    ensure_manifest(
        log_path,
        {"benchmark": benchmark, "models": models, "strategies": strategies,
         "variants": variants, "temperature": 0.0},
        input_files,
    )
    if result_path.exists() and not log_path.exists():
        raise FileExistsError(f"{result_path} existiert ohne Fortschrittsdatei. Bitte zuerst sichern.")

    dataset = load_jsonl(data_file)
    rows = load_progress(log_path)
    done = completed_keys(rows)
    consecutive_errors = 0
    print(f"Sensitivity {benchmark}: {len(rows)} bisher gespeicherte Aufrufe")

    try:
        for model in models:
            for strategy in strategies:
                for variant in variants:
                    template_path = (
                        Path("prompt_templates") / benchmark / "sensitivity"
                        / strategy / f"{variant}.txt"
                    )
                    template = template_path.read_text(encoding="utf-8")

                    for item_index, item in enumerate(dataset):
                        key = (benchmark, model, strategy, variant, item_index)
                        if key in done:
                            continue

                        if benchmark == "mmlu_pro":
                            prompt = format_mmlu(template, item)
                            solution = item["answer"]
                        else:
                            prompt = format_gsm8k(template, item)
                            solution = extract_ground_truth(item["answer"])
                            if solution is None:
                                raise ValueError(f"GSM8K-Ground-Truth fehlt bei Aufgabe {item_index}")

                        result = query_ollama(model, prompt)
                        response_text = result.get("response", "")
                        predicted = (
                            extract_mmlu(response_text) if benchmark == "mmlu_pro"
                            else extract_gsm8k(response_text)
                        )
                        is_correct = (
                            predicted == solution if benchmark == "mmlu_pro"
                            else answers_equal(predicted, solution)
                        )
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
                        r for r in rows.values() if r["model"] == model
                        and r["strategy"] == strategy and r["variant"] == variant
                    ]
                    correct = sum(r["is_correct"] for r in subset)
                    print(f"{benchmark} | {model} | {strategy} | {variant}: "
                          f"{correct}/{len(dataset)} richtig; {len(subset)} erfasst")
        failures = sum(row.get("error") is not None for row in rows.values())
        if failures:
            raise RuntimeError(
                f"Noch {failures} Ollama-Fehler für {benchmark}. "
                "Starte denselben Befehl erneut; nur fehlerhafte Aufrufe werden wiederholt."
            )
    finally:
        save_final(result_path, rows)
        print(f"Aktueller Stand: {len(rows)} Ergebnisse in {result_path}")

    first = Path("results/sensitivity_mmlu_pro.json")
    second = Path("results/sensitivity_gsm8k.json")
    if first.exists() and second.exists():
        combined = load_json_file(first) + load_json_file(second)
        save_json_atomic("results/sensitivity_results.json", combined)
        print("Beide Benchmarks kombiniert: results/sensitivity_results.json")


def load_json_file(path):
    import json
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["mmlu_pro", "gsm8k"])
    args = parser.parse_args()
    run(args.benchmark)
