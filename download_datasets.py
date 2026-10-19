"""Lädt die Datensätze für die Evaluierung der Benchmarks MMLU-Pro und GSM8K über HuggingFace herunter.

Erstellt je Datensatz eine Stichprobe aus 50 zufällig ausgewählten Datensätzen und speichert diese als JSONL-Datei 
für die folgende Pipeline-Verarbeitung ab."""

from pathlib import Path
import json
import random
from datasets import load_dataset

# Pfadkonfiguration für die Speicherung der Datensätze
file_path = Path(r"C:\Users\hasan\OneDrive\Desktop\bachelor_labor") / "data"
file_path.mkdir(parents=True, exist_ok=True)

# Ausgewählter Seed-Wert für die Reproduzierbarkeit
random.seed(42)

def save_jsonl(file_path, data_list):
    """Speichert eine Liste von Datensätzen im JSONL-Format in einer Datei.
    
    Args:
        file_path (Path): Zielpfad für die JSONL-Datei.
        data_list (list[dict]): Liste der zu speichernden Datensätze.
        
    Returns:
        None: Daten werden direkt auf die Festplatte geschrieben.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for data in data_list:
            file.write(json.dumps(data) + "\n")

def mmlu_pro_test(sample_size = 50):
    """Lädt für die Stichprobe MMLU-Pro einen Testdatensatz von HuggingFace runter.
    
    Args:
        sample_size (int): Größe der Stichprobe.
        
    Returns:
        None: Datensatz wird direkt auf der Festplatte gespeichert.
    """
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split = "test")
    samples = random.sample(range(len(dataset)), sample_size)
    selected_samples = []
    for i in samples:
        selected_samples.append(dataset[i])
    output_path = file_path / "mmlu_pro_samples50.jsonl"
    save_jsonl(output_path, selected_samples)

def gsm8k_test(sample_size = 50):
   """Lädt für die Stichprobe gsm8k einen Testdatensatz von HuggingFace runter.
       
    Args:
        sample_size (int): Größe der Stichprobe.
           
    Returns:
        None: Datensatz wird direkt auf der Festplatte gespeichert.
    """
   dataset = load_dataset("openai/gsm8k", "main", split = "test")
   samples = random.sample(range(len(dataset)), sample_size)
   selected_samples = []
   for i in samples:
       selected_samples.append(dataset[i])
   output_path = file_path / "gsm8k_samples50.jsonl"
   save_jsonl(output_path, selected_samples)

mmlu_pro_test()
gsm8k_test()
    