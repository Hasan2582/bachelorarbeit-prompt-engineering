from pathlib import Path

from checkpoint import ensure_manifest, load_jsonl, load_progress, save_final
from run_gsm8k import run_gsm8k
from run_mmlu import run_mmlu


models = ["llama3:8b", "phi3:mini"]
strategies = ["zero_shot", "one_shot", "few_shot", "cot", "cod", "role_prompting"]

mmlu_file = Path("data/mmlu_pro_samples50.jsonl")
gsm8k_file = Path("data/gsm8k_samples50.jsonl")
log_path = Path("results/main_progress.jsonl")
result_path = Path("results/eval_run.json")


if __name__ == "__main__":
    input_files = [mmlu_file, gsm8k_file] + [
        Path("prompt_templates") / benchmark / f"{strategy}.txt"
        for benchmark in ("mmlu_pro", "gsm8k")
        for strategy in strategies
    ]
    ensure_manifest(
        log_path,
        {"models": models, "strategies": strategies, "temperature": 0.0},
        input_files,
    )
    if result_path.exists() and not log_path.exists():
        raise FileExistsError("eval_run.json existiert ohne Fortschrittsdatei. Bitte zuerst sichern.")

    rows = load_progress(log_path)
    print(f"Hauptversuch: {len(rows)} bisher gespeicherte Aufrufe")
    mmlu_dataset = load_jsonl(mmlu_file)
    gsm8k_dataset = load_jsonl(gsm8k_file)

    try:
        run_mmlu(models, strategies, mmlu_dataset, rows, log_path)
        run_gsm8k(models, strategies, gsm8k_dataset, rows, log_path)
        failures = sum(row.get("error") is not None for row in rows.values())
        if failures:
            raise RuntimeError(
                f"Noch {failures} Ollama-Fehler im Hauptversuch. "
                "Starte denselben Befehl erneut; nur fehlerhafte Aufrufe werden wiederholt."
            )
    finally:
        save_final(result_path, rows)
        print(f"Aktueller Stand: {len(rows)} Ergebnisse in {result_path}")
