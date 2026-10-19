from pathlib import Path
import hashlib
import json
import os
import shutil


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def record_key(row):
    return (
        row["benchmark"],
        row["model"],
        row["strategy"],
        row.get("variant"),
        row["item_index"],
    )


def load_progress(log_path):
    rows = {}
    log_path = Path(log_path)
    if not log_path.exists():
        return rows

    for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows[record_key(row)] = row
        except (ValueError, KeyError) as error:
            raise ValueError(
                f"Ungültiger Checkpoint in {log_path}, Zeile {line_number}. "
                "Sichere die Datei und korrigiere die betroffene Zeile."
            ) from error
    return rows


def completed_keys(rows):
    return {key for key, row in rows.items() if row.get("error") is None}


def append_result(log_path, rows, row):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    rows[record_key(row)] = row


def save_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def save_final(path, rows):
    save_json_atomic(path, list(rows.values()))


def ensure_manifest(log_path, config, input_files):
    log_path = Path(log_path)
    manifest_path = log_path.with_name(log_path.stem + "_manifest.json")
    inputs = {
        Path(path).as_posix(): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in input_files
    }
    current = {"config": config, "sha256_inputs": inputs}

    if not manifest_path.exists():
        if log_path.exists() and log_path.stat().st_size > 0:
            raise ValueError(
                f"Checkpoint ohne Manifest gefunden: {log_path}. "
                "Alte Datei sichern/umbenennen, bevor du einen neuen Versuch beginnst."
            )
        save_json_atomic(manifest_path, current)
        return

    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous == current:
        return

    old_inputs = previous.get("sha256_inputs", {})
    if previous.get("config") != config or set(old_inputs) != set(inputs):
        raise ValueError(
            "Konfiguration oder Eingabedateien haben sich geändert. "
            "Bestehenden Lauf nicht mit einem anderen Versuch vermischen!"
        )

    rows = load_progress(log_path)
    changed = [path for path in inputs if old_inputs[path] != inputs[path]]
    if not changed:
        raise ValueError("Unbekannte Änderung am Manifest. Bitte prüfen.")

    for path in changed:
        parts = Path(path).parts
        if "prompt_templates" in parts:
            parts = parts[parts.index("prompt_templates"):]
        elif "data" in parts:
            parts = parts[parts.index("data"):]
        affected = True

        if len(parts) == 3 and parts[0] == "prompt_templates":
            benchmark, strategy = parts[1], Path(parts[2]).stem
            affected = any(
                row.get("benchmark") == benchmark and row.get("strategy") == strategy
                for row in rows.values()
            )
        elif (len(parts) == 5 and parts[0] == "prompt_templates"
              and parts[2] == "sensitivity"):
            benchmark, strategy, variant = parts[1], parts[3], Path(parts[4]).stem
            affected = any(
                row.get("benchmark") == benchmark
                and row.get("strategy") == strategy
                and row.get("variant") == variant
                for row in rows.values()
            )
        elif len(parts) == 2 and parts[0] == "data":
            filename = parts[1]
            benchmark = (
                "mmlu_pro" if filename.startswith("mmlu_pro_") else
                "gsm8k" if filename.startswith("gsm8k_") else None
            )
            affected = benchmark is None or any(
                row.get("benchmark") == benchmark for row in rows.values()
            )

        if affected:
            raise ValueError(
                f"{path} wurde verändert, aber für die betroffene Gruppe existieren "
                "bereits Ergebnisse. Datei zurücksetzen oder separaten Versuch beginnen."
            )

    backup_path = manifest_path.with_name(manifest_path.name + ".bak")
    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)
    save_json_atomic(manifest_path, current)
    print("Manifest aktualisiert: nur bisher ungetestete Eingaben wurden korrigiert.")
    for path in changed:
        print(f"  - {path}")
