import sys
import os

class LLMRouterClient:
    print("Initializing LLMRouterClient...")
    def __init__(self, strong_model="openai/big-model", weak_model="ollama/qwen2.5:3b", router_type="bert", threshold=0.5, api_base="http://localhost:8000/v1"):
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-local-routing"
        if api_base:
            os.environ["OPENAI_API_BASE"] = api_base
            
        repo_path = r"RouteLLM"
        if repo_path not in sys.path:
            sys.path.append(repo_path)
            
        from routellm.controller import Controller
        
        self.router_model_name = f"router-{router_type}-{threshold}"
        self.client = Controller(
            routers=[router_type],
            strong_model=strong_model,
            weak_model=weak_model
        )

    def generate_response(self, prompt: str):
        print(f"Generating response for prompt: {prompt}")
        response = self.client.chat.completions.create(
            model=self.router_model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"Generated response: {response.choices[0].message.content}")
        return {
            "routed_to": response.model,
            "content": response.choices[0].message.content,
            "raw_response": response
        }

