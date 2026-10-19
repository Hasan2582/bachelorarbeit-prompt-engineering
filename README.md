# Bachelorarbeit – Prompt Engineering
Dieses Repository enthält den Python-Code, die Prompt-Templates und die Ergebnisse der Experimente zur Untersuchung verschiedener Prompting-Strategien bei lokalen Sprachmodellen.

# Modelle und Benchmarks
- Modelle: Llama 3 8B und Phi-3 Mini
- Benchmarks: GSM8K und MMLU-Pro
- Stichprobe: jeweils 50 Aufgaben
- Ausführung: lokal über Ollama

# Untersuchte Strategien
Zero-Shot, One-Shot, Few-Shot, Chain-of-Thought (CoT), Chain-of-Draft (CoD) und Role Prompting.
Zusätzlich wird die Prompt-Sensitivität von Zero-Shot und Few-Shot anhand von fünf Varianten untersucht: Baseline, Character, Word, Sentence und Semantic.

# Projektstruktur
bachelor_labor/
├── data/                 
├── prompt_templates/     
├── results/              
├── main_pipeline.py      
├── run_gsm8k.py
├── run_mmlu.py
├── run_sensitivity.py
├── ollama_client.py
├── checkpoint.py
├── download_datasets.py
├── analyze_results.py
├── analyze_sensitivity.py
├── audit_sensitivity.py
├── prepare_thesis_results.py
└── README.md

# Durchführung
Voraussetzungen: Python, Ollama sowie `requests`, `datasets`, `pandas` und `matplotlib`.

Hauptversuch:
```bash
python main_pipeline.py
```

Sensitivitätsversuch:
```bash
python run_sensitivity.py mmlu_pro
python run_sensitivity.py gsm8k
```


Auswertung:
```bash
python analyze_results.py
python analyze_sensitivity.py
python audit_sensitivity.py
python prepare_thesis_results.py
```
