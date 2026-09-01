from router import LLMRouterClient
from prompt_compresstion import PromptCompressorClient

prompt= "An investor purchases a 10-year government bond with a face value of ₹10,000 at a price of ₹9,600. The bond pays an annual coupon of 7% on face value. After one year, the bond is sold for ₹9,850. If inflation during the year was 5.5%, calculate the investor’s nominal holding-period return and approximate real return."
prompt_compressor = PromptCompressorClient()
og_token_count = prompt_compressor.count_tokens(prompt)
print(f"Original prompt token count: {og_token_count}")
compressed_prompt, compression_time = prompt_compressor.compress_prompt(prompt, target_rate=0.7)
print(f"Prompt compressed in {compression_time:.2f} ms")
print("="*50)
print(f"Original Prompt: {prompt}")
print("="*50)
comp_token_count = prompt_compressor.count_tokens(compressed_prompt)
print(f"Compressed prompt token count: {comp_token_count}")
print("="*50)

router_client = LLMRouterClient(strong_model="ollama/gpt-oss:120b-cloud", weak_model="ollama/qwen2.5:3b")
result = router_client.generate_response(prompt)
print("="*50)
print(result["routed_to"])
print("="*50)
      
print(result["routed_to"], result["content"][:100])
old_model = result["routed_to"]






# 4. Route the prompt and generate the response
result = router_client.generate_response(compressed_prompt)

new_model = result["routed_to"]

print("="*50)
print(f"old model: {old_model} | new model: {new_model}")
