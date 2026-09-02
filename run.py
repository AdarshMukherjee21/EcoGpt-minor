from config import settings
from router import LLMRouterClient
from prompt_compression import PromptCompressorClient
from green_metric import BlackBoxLLMEstimator
from raw_model_runner import RawModelRunner

# 1. Initialize Clients & Estimator (all from config)
prompt_compressor = PromptCompressorClient()
router_client = LLMRouterClient()
eco_estimator = BlackBoxLLMEstimator()
raw_runner = RawModelRunner()

SMALL = settings.models.small
LARGE = settings.models.large

prompt = (
    "An investor purchases a 10-year government bond with a face value of ₹10,000 "
    "at a price of ₹9,600. The bond pays an annual coupon of 7% on face value. "
    "After one year, the bond is sold for ₹9,850. If inflation during the year was 5.5%, "
    "calculate the investor’s nominal holding-period return and approximate real return."
)

# 2. Compress Prompt
og_token_count = prompt_compressor.count_tokens(prompt)
print(f"Original prompt token count: {og_token_count}")

compressed_prompt, compression_time = prompt_compressor.compress_prompt(
    prompt, target_rate=0.7
)
comp_token_count = prompt_compressor.count_tokens(compressed_prompt)

print(f"Prompt compressed in {compression_time:.2f} ms")
print("=" * 50)
print(f"Original Prompt: {prompt}")
print("=" * 50)
print(f"Compressed prompt token count: {comp_token_count}")
print("=" * 50)

# 3. Route & Run Original Prompt
orig_result = router_client.generate_response(prompt)
old_model = orig_result["routed_to"]
orig_output_tokens = prompt_compressor.count_tokens(orig_result["content"])

print(f"Original Run Routed To: {old_model}")
print(f"Sample Content: {orig_result['content'][:100]}...")
print("=" * 50)

# 4. Route & Run Compressed Prompt
comp_result = router_client.generate_response(compressed_prompt)
new_model = comp_result["routed_to"]
comp_output_tokens = prompt_compressor.count_tokens(comp_result["content"])

print(f"Compressed Run Routed To: {new_model}")
print(f"Sample Content: {comp_result['content'][:100]}...")
print("=" * 50)
print(f"Model Transition -> Old model: {old_model} | New model: {new_model}")
print("=" * 50)

# 5. Calculate Green Metrics (Energy & Carbon)
orig_green = eco_estimator.estimate(
    model_tag=old_model, input_tokens=og_token_count, output_tokens=orig_output_tokens
)

comp_green = eco_estimator.estimate(
    model_tag=new_model, input_tokens=comp_token_count, output_tokens=comp_output_tokens
)

# Extract P50 (median) estimates
orig_energy_p50 = orig_green["energy_wh"]["P50"]
comp_energy_p50 = comp_green["energy_wh"]["P50"]

orig_carbon_p50 = orig_green["carbon_gco2e"]["P50"]
comp_carbon_p50 = comp_green["carbon_gco2e"]["P50"]

orig_water_p50 = orig_green["water_liters"]["P50"]
comp_water_p50 = comp_green["water_liters"]["P50"]

# Calculate Savings
energy_saved_pct = ((orig_energy_p50 - comp_energy_p50) / orig_energy_p50) * 100
carbon_saved_pct = ((orig_carbon_p50 - comp_carbon_p50) / orig_carbon_p50) * 100
water_saved_pct = ((orig_water_p50 - comp_water_p50) / orig_water_p50) * 100

# 6. Print Environmental Impact Comparison
print("\nENVIRONMENTAL IMPACT & GREEN METRICS REPORT")
print("=" * 50)
print(f"{'Metric':<25} | {'Original Run':<15} | {'Compressed Run':<15}")
print("-" * 60)
print(f"{'Routed Model':<25} | {old_model:<15} | {new_model:<15}")
print(f"{'Input Tokens':<25} | {og_token_count:<15} | {comp_token_count:<15}")
print(f"{'Output Tokens':<25} | {orig_output_tokens:<15} | {comp_output_tokens:<15}")
print(
    f"{'Energy (P50 Median)':<25} | {orig_energy_p50:.6f} Wh   | {comp_energy_p50:.6f} Wh"
)
print(
    f"{'Carbon (P50 Median)':<25} | {orig_carbon_p50:.6f} gCO2e| {comp_carbon_p50:.6f} gCO2e"
)
print(f"{'Water (P50 Median)':<25} | {orig_water_p50:.6f} L   | {comp_water_p50:.6f} L")
print("=" * 50)
print(f"Energy Savings:  {energy_saved_pct:.2f}%")
print(f"Carbon Savings:  {carbon_saved_pct:.2f}%")
print(f"Water Savings:   {water_saved_pct:.2f}%")
