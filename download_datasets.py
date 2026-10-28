
"""Lädt die Benchmarks MMLU-Pro und GSM8K über Hugging Face.
Wählt je Benchmark 50 zufällige Aufgaben aus.
Speichert die Stichproben als JSONL-Dateien."""

from pathlib import Path
import json
import random
from datasets import load_dataset


#Pfad für die Datensätze
file_path = Path(__file__).resolve().parent / "data"
file_path.mkdir(parents=True, exist_ok=True)


#Seed für die Zufallsauswahl
random.seed(42)


def save_jsonl(file_path, data_list):
    """Speichert die ausgewählten Aufgaben im JSONL-Format.

    Args:
        file_path (Path): Pfad zur JSONL-Datei
        data_list (list[dict]): Liste der ausgewählten Aufgaben

    Returns:
        None: Daten werden in der Datei gespeichert
    """
    with open(file_path, "w", encoding="utf-8") as file:
        for data in data_list:
            file.write(json.dumps(data) + "\n")


def mmlu_pro_test(sample_size=50):
    """Lädt und speichert eine Stichprobe aus MMLU-Pro.

    Args:
        sample_size (int): Anzahl der Aufgaben

    Returns:
        None: Stichprobe wird als JSONL-Datei gespeichert
    """
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    samples = random.sample(range(len(dataset)), sample_size)

    #Auswahl der Aufgaben
    selected_samples = []
    for i in samples:
        selected_samples.append(dataset[i])

    #Speicherung der Stichprobe
    output_path = file_path / "mmlu_pro_samples50.jsonl"
    save_jsonl(output_path, selected_samples)


def gsm8k_test(sample_size=50):
    """Lädt und speichert eine Stichprobe aus GSM8K.

    Args:
        sample_size (int): Anzahl der Aufgaben

    Returns:
        None: Stichprobe wird als JSONL-Datei gespeichert
    """
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    samples = random.sample(range(len(dataset)), sample_size)

    #Auswahl der Aufgaben
    selected_samples = []
    for i in samples:
        selected_samples.append(dataset[i])

    #Speicherung der Stichprobe
    output_path = file_path / "gsm8k_samples50.jsonl"
    save_jsonl(output_path, selected_samples)


#Erstellung der beiden Stichproben
mmlu_pro_test()
gsm8k_test()
