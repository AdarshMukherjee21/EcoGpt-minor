import requests

class RawModelRunner:
    """
    Direct interface for running Ollama models via Ollama's local REST API.
    """
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host.rstrip('/')

    def run_model(self, model: str, prompt: str) -> str:
        """
        Executes a prompt on a specified Ollama model and returns the text response.
        """
        # Strip provider prefix if present (e.g., "ollama/qwen2.5:3b" -> "qwen2.5:3b")
        clean_model = model.split('/')[-1] if '/' in model else model

        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": clean_model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed for model '{clean_model}': {str(e)}")