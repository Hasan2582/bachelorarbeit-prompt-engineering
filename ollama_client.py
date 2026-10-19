import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

def query_ollama(model_name, prompt):
    start_time = time.perf_counter()

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
        response = requests.post(
            OLLAMA_URL,
            json=ollama_informations,
            timeout=900
        )

        response.raise_for_status()

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
        latency = time.perf_counter() - start_time

        return {
            "response": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": latency,
            "error": str(error)
        }
    

    # response = requests.post(OLLAMA_URL, json=ollama_informations)
    # latency = time.perf_counter() - start_time
    # data = response.json()
    # response_text = data.get("response", "")
    # input_tokens = data.get("prompt_eval_count", 0)
    # output_tokens = data.get("eval_count", 0)

    # result = {
    #     "response": response_text,
    #     "input_tokens": input_tokens,
    #     "output_tokens": output_tokens,
    #     "time": latency
    # }

    # return result
