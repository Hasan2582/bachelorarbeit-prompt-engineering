
"""Speicherung des Versuchsfortschritts und Sicherung der Eingabedaten.
Fortsetzung von unterbrochenen Versuchen.
Prüfung der Versuchskonfiguration und verwendeten Dateien."""

from pathlib import Path
import hashlib
import json
import os
import shutil


def load_jsonl(path):
    """Einlesen der gespeicherten Einträge einer JSONL-Datei.

    Args:
        path (Path): Pfad zur JSONL-Datei

    Returns:
        list: Liste der eingelesenen Einträge
    """
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def record_key(row):
    """Erstellung eines Schlüssels für einen Modellaufruf.

    Args:
        row (dict): Ergebnis eines Modellaufrufs

    Returns:
        tuple: Schlüssel aus Benchmark, Modell, Strategie, Variante und Aufgabenindex
    """
    return (
        row["benchmark"],
        row["model"],
        row["strategy"],
        row.get("variant"),
        row["item_index"],
    )


def load_progress(log_path):
    """Einlesen der bereits gespeicherten Versuchsergebnisse.

    Args:
        log_path (Path): Pfad zur Checkpoint-Datei

    Returns:
        dict: Gespeicherte Ergebnisse mit ihren Schlüsseln
    """
    rows = {}
    log_path = Path(log_path)

    if not log_path.exists():
        return rows

    #Einlesen der einzelnen Zeilen aus dem Checkpoint
    for line_number, line in enumerate(
        log_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue

        try:
            row = json.loads(line)
            rows[record_key(row)] = row
        except (ValueError, KeyError) as error:
            raise ValueError(
                f"Fehler in {log_path}, Zeile {line_number}"
            ) from error

    return rows


def completed_keys(rows):
    """Ermittlung der bereits abgeschlossenen Modellaufrufe.

    Args:
        rows (dict): Bisher gespeicherte Versuchsergebnisse

    Returns:
        set: Schlüssel der Modellaufrufe ohne technische Fehler
    """
    return {key for key, row in rows.items() if row.get("error") is None}


def append_result(log_path, rows, row):
    """Speicherung eines neuen Modellaufrufs im Checkpoint.

    Args:
        log_path (Path): Pfad zur Checkpoint-Datei
        rows (dict): Bisher gespeicherte Ergebnisse
        row (dict): Neues Ergebnis eines Modellaufrufs

    Returns:
        None: Ergebnis wird in der Datei und im Dictionary gespeichert
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    #Ergebnis an die Datei anhängen
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())

    rows[record_key(row)] = row


def save_json_atomic(path, data):
    """Speicherung von JSON-Daten über eine temporäre Datei.

    Args:
        path (Path): Pfad zur endgültigen JSON-Datei
        data (list | dict): Zu speichernde Daten

    Returns:
        None: Daten werden als JSON-Datei gespeichert
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    #Speicherung in einer temporären Datei
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())

    #Temporäre Datei ersetzen
    os.replace(temporary, path)


def save_final(path, rows):
    """Speicherung aller Versuchsergebnisse als JSON-Datei.

    Args:
        path (Path): Pfad zur Ergebnisdatei
        rows (dict): Gesammelte Versuchsergebnisse

    Returns:
        None: Ergebnisse werden in einer JSON-Datei gespeichert
    """
    save_json_atomic(path, list(rows.values()))


def ensure_manifest(log_path, config, input_files):
    """Überprüfung der Konfiguration und Eingabedateien eines Versuchs.

    Args:
        log_path (Path): Pfad zur Checkpoint-Datei
        config (dict): Einstellungen für den Versuch
        input_files (list): Pfade der verwendeten Eingabedateien

    Returns:
        None: Konfigurationsdatei wird überprüft oder gespeichert
    """
    log_path = Path(log_path)
    manifest_path = log_path.with_name(log_path.stem + "_manifest.json")

    #Prüfsummen der Eingabedateien
    inputs = {
        Path(path).as_posix(): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in input_files
    }
    current = {"config": config, "sha256_inputs": inputs}

    #Prüfen, ob die Konfigurationsdatei vorhanden ist
    if not manifest_path.exists():
        if log_path.exists() and log_path.stat().st_size > 0:
            raise ValueError(
                f"Konfigurationsdatei fehlt: {log_path}"
            )
        save_json_atomic(manifest_path, current)
        return

    #Vergleich mit den gespeicherten Einstellungen
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous == current:
        return

    old_inputs = previous.get("sha256_inputs", {})

    if previous.get("config") != config or set(old_inputs) != set(inputs):
        raise ValueError("Konfiguration oder Eingabedateien geändert")

    #Ermittlung der veränderten Eingabedateien
    rows = load_progress(log_path)
    changed = [path for path in inputs if old_inputs[path] != inputs[path]]

    if not changed:
        raise ValueError("Konfigurationsdatei stimmt nicht überein")

    #Prüfen, ob die Dateien schon verwendet wurden
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
                row.get("benchmark") == benchmark
                and row.get("strategy") == strategy
                for row in rows.values()
            )

        elif (
            len(parts) == 5
            and parts[0] == "prompt_templates"
            and parts[2] == "sensitivity"
        ):
            benchmark, strategy, variant = (
                parts[1], parts[3], Path(parts[4]).stem
            )
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
                row.get("benchmark") == benchmark
                for row in rows.values()
            )

        if affected:
            raise ValueError(f"Datei bereits verwendet: {path}")

    #Konfigurationsdatei aktualisieren
    backup_path = manifest_path.with_name(manifest_path.name + ".bak")

    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)

    save_json_atomic(manifest_path, current)

    print("Konfiguration aktualisiert")
    for path in changed:
        print(path)
