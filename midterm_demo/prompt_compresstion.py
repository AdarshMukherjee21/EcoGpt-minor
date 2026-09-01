import time
from llmlingua import PromptCompressor
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

class PromptCompressorClient:
    def __init__(
        self, 
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", 
        device_map: str = "cpu"
    ):
        print("Loading LLMLingua-2 model...")
        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device_map  # Forces PyTorch to load and run the BERT model on CPU
        )
        print("Compressor successfully loaded!")

    def compress_prompt(self, prompt: str, target_rate: float = 0.5) -> tuple[str, float]:
        """Compresses prompt using LLMLingua-2 and measures compression latency."""
        start_time = time.perf_counter()

        results = self.compressor.compress_prompt_llmlingua2(
            prompt,
            rate=target_rate,
            force_tokens=['\n', '.', '!', '?', ','],
            drop_consecutive=True
        )

        end_time = time.perf_counter()
        comp_time_ms = (end_time - start_time) * 1000

        return results['compressed_prompt'], comp_time_ms

    

    def count_tokens(self,text: str) -> int:
        return len(tokenizer.encode(text))