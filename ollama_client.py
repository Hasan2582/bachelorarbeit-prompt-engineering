
"""Kommunikation mit der lokalen Ollama-API.
Senden der Prompts und Empfangen der Modellantworten.
Erfassung von Tokenverbrauch, Latenz und technischen Fehlern."""

import requests
import time


#Adresse der lokalen Ollama-API
OLLAMA_URL = "http://localhost:11434/api/generate"


def query_ollama(model_name, prompt):
    """Senden eines Prompts an Ollama und Rückgabe der Ergebnisse.

    Args:
        model_name (str): Name des verwendeten Sprachmodells
        prompt (str): Prompt für das Modell

    Returns:
        dict: Modellantwort, Tokenverbrauch, Latenz und mögliche Fehler
    """
    start_time = time.perf_counter()

    #Einstellungen für die Modellanfrage
    ollama_informations = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 512
        }
    }

    try:
        #Anfrage an Ollama senden
        response = requests.post(
            OLLAMA_URL,
            json=ollama_informations,
            timeout=900
        )
        response.raise_for_status()

        #Antwort und Messwerte auslesen
        latency = time.perf_counter() - start_time
        data = response.json()

        return {
            "response": data.get("response", ""),
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "latency": latency,
            "error": None
        }

    except Exception as error:
        #Fehler und Antwortzeit erfassen
        latency = time.perf_counter() - start_time

        return {
            "response": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": latency,
            "error": str(error)
        }
