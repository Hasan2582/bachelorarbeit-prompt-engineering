
# Bachelorarbeit – Prompt Engineering

Dieses Repository enthält den Python-Code, die Prompt-Templates und die Ergebnisse der Experimente zur Untersuchung verschiedener Prompting-Strategien bei lokalen Sprachmodellen.

## Modelle und Benchmarks

- Modelle: Llama 3 8B und Phi-3 Mini
- Benchmarks: GSM8K und MMLU-Pro
- Stichprobe: jeweils 50 Aufgaben
- Ausführung: lokal über Ollama

## Untersuchte Strategien

Im Hauptversuch werden sechs Strategien untersucht: Zero-Shot, One-Shot, Few-Shot, Chain-of-Thought (CoT), Chain-of-Draft (CoD) und Role Prompting.

Im Sensitivitätsversuch werden Zero-Shot und Few-Shot mit fünf Promptvarianten untersucht: Baseline, Character, Word, Sentence und Semantic.

## Projektstruktur

```text
bachelor_labor/
├── data/
├── prompt_templates/
├── results/
├── main_experiment.py
├── main_analysis.py
├── sensitivity_experiment.py
├── sensitivity_analysis.py
├── sensitivity_quality_control.py
├── run_gsm8k.py
├── run_mmlu.py
├── ollama_client.py
├── checkpoint.py
├── download_datasets.py
├── result_tables.py
└── README.md
```

## Durchführung

Voraussetzungen: Python, Ollama sowie `requests`, `datasets`, `pandas` und `matplotlib`.

Datensätze herunterladen:

```bash
python download_datasets.py
```

Hauptversuch:

```bash
python main_experiment.py
```

Sensitivitätsversuch:

```bash
python sensitivity_experiment.py mmlu_pro
python sensitivity_experiment.py gsm8k
```

Auswertung:

```bash
python main_analysis.py
python sensitivity_analysis.py
python sensitivity_quality_control.py
python result_tables.py
```

Die Ergebnisse werden im Ordner `results` gespeichert.
