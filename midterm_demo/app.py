import warnings
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from router import LLMRouterClient
from prompt_compresstion import PromptCompressorClient
from green_metric import BlackBoxLLMEstimator
from raw_model_runner import RawModelRunner

# Suppress pandas SettingWithCopyWarning coming from green_metric.py
warnings.simplefilter(action='ignore', category=FutureWarning)
try:
    pd.options.mode.chained_assignment = None
except Exception:
    pass

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins and API endpoints

# Define model identifiers
STRONG_MODEL = "ollama/gpt-oss:120b-cloud"
WEAK_MODEL = "ollama/qwen2.5:3b"

# Initialize singletons on server start
prompt_compressor = PromptCompressorClient()
router_client = LLMRouterClient(
    strong_model=STRONG_MODEL, 
    weak_model=WEAK_MODEL
)
eco_estimator = BlackBoxLLMEstimator()
raw_runner = RawModelRunner()


class PromptFormatter:
    """
    Utility class to format input prompts and sanitize raw LLM responses
    to prevent JSON serialization and string control character failures.
    """
    @staticmethod
    def format_input(prompt: str, instruction: str = None) -> str:
        cleaned = prompt.strip()
        if instruction:
            return f"{instruction}\n\nTask:\n{cleaned}"
        return cleaned

    @staticmethod
    def clean_output(text: str) -> str:
        if not isinstance(text, str):
            return str(text)
        # Remove null characters and standard string corruption bits
        cleaned_text = text.replace("\x00", "").strip()
        return cleaned_text


@app.route("/api/router", methods=["POST"])
def route_prompt():
    """
    Endpoint 1: Router
    Routes prompt. If routed to the small model, executes comparative
    large model run and returns green metric savings.
    """
    data = request.get_json() or {}
    raw_prompt = data.get("prompt")
    
    if not raw_prompt:
        return jsonify({"error": "Field 'prompt' is required."}), 400

    # Format incoming prompt
    formatted_prompt = PromptFormatter.format_input(raw_prompt)

    try:
        # 1. Route prompt and get primary response
        routed_result = router_client.generate_response(formatted_prompt)
        routed_to = routed_result.get("routed_to", WEAK_MODEL)
        routed_content = PromptFormatter.clean_output(routed_result.get("content", ""))

        # 2. Token counts for routed execution
        input_tokens = prompt_compressor.count_tokens(formatted_prompt)
        routed_output_tokens = prompt_compressor.count_tokens(routed_content)

        # 3. Safe calculation of Green metrics for routed model
        try:
            routed_green = eco_estimator.estimate(
                model_tag=routed_to,
                input_tokens=input_tokens,
                output_tokens=routed_output_tokens
            )
        except Exception as green_err:
            routed_green = {"error": f"Failed to compute green metrics: {str(green_err)}"}

        response_payload = {
            "routed_to": routed_to,
            "content": routed_content,
            "input_tokens": input_tokens,
            "output_tokens": routed_output_tokens,
            "green_metrics": routed_green,
            "saved_by_routing": False
        }

        # 4. Check if routed to the small model
        is_small_model = (routed_to == WEAK_MODEL) or ("qwen" in str(routed_to).lower()) or ("3b" in str(routed_to).lower())
        
        if is_small_model:
            try:
                # Run prompt through large model for comparison
                large_raw_content = raw_runner.run_model(model=STRONG_MODEL, prompt=formatted_prompt)
                large_content = PromptFormatter.clean_output(large_raw_content)
                large_output_tokens = prompt_compressor.count_tokens(large_content)
                
                large_green = eco_estimator.estimate(
                    model_tag=STRONG_MODEL,
                    input_tokens=input_tokens,
                    output_tokens=large_output_tokens
                )

                small_energy = routed_green.get("energy_wh", {}).get("P50", 0)
                large_energy = large_green.get("energy_wh", {}).get("P50", 0)
                small_carbon = routed_green.get("carbon_gco2e", {}).get("P50", 0)
                large_carbon = large_green.get("carbon_gco2e", {}).get("P50", 0)

                energy_saved_pct = ((large_energy - small_energy) / large_energy * 100) if large_energy > 0 else 0
                carbon_saved_pct = ((large_carbon - small_carbon) / large_carbon * 100) if large_carbon > 0 else 0

                response_payload["saved_by_routing"] = True
                response_payload["large_model_comparison"] = {
                    "large_model": STRONG_MODEL,
                    "large_model_content": large_content,
                    "large_model_output_tokens": large_output_tokens,
                    "large_model_green_metrics": large_green,
                    "energy_saved_wh_p50": round(large_energy - small_energy, 6),
                    "carbon_saved_gco2e_p50": round(large_carbon - small_carbon, 6),
                    "energy_saved_pct": round(energy_saved_pct, 2),
                    "carbon_saved_pct": round(carbon_saved_pct, 2)
                }
            except Exception as raw_err:
                response_payload["large_model_comparison_error"] = str(raw_err)

        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/raw-runner", methods=["POST"])
def run_raw_model():
    """
    Endpoint 2: Raw Model Runner
    """
    data = request.get_json() or {}
    model = data.get("model")
    prompt = data.get("prompt")

    if not model or not prompt:
        return jsonify({"error": "Fields 'model' and 'prompt' are required."}), 400

    try:
        formatted_prompt = PromptFormatter.format_input(prompt)
        response_text = raw_runner.run_model(model=model, prompt=formatted_prompt)
        cleaned_response = PromptFormatter.clean_output(response_text)

        return jsonify({
            "model": model,
            "content": cleaned_response
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compress", methods=["POST"])
def compress_prompt():
    """
    Endpoint 3: Prompt Compressor with Pre/Post Green Metrics
    """
    data = request.get_json() or {}
    prompt = data.get("prompt")
    target_rate = data.get("target_rate", 0.7)
    model = data.get("model", WEAK_MODEL)

    if not prompt:
        return jsonify({"error": "Field 'prompt' is required."}), 400

    try:
        formatted_prompt = PromptFormatter.format_input(prompt)

        # 1. Token count & Compress Prompt
        og_token_count = prompt_compressor.count_tokens(formatted_prompt)
        compressed_prompt, compression_time = prompt_compressor.compress_prompt(
            formatted_prompt, target_rate=target_rate
        )
        comp_token_count = prompt_compressor.count_tokens(compressed_prompt)

        # 2. Run raw model on original prompt
        try:
            og_content = PromptFormatter.clean_output(raw_runner.run_model(model=model, prompt=formatted_prompt))
            og_output_tokens = prompt_compressor.count_tokens(og_content)
        except Exception:
            og_content = ""
            og_output_tokens = 0

        # 3. Run raw model on compressed prompt
        try:
            comp_content = PromptFormatter.clean_output(raw_runner.run_model(model=model, prompt=compressed_prompt))
            comp_output_tokens = prompt_compressor.count_tokens(comp_content)
        except Exception:
            comp_content = ""
            comp_output_tokens = 0

        # 4. Calculate Green Metrics Pre-Compression
        pre_green = eco_estimator.estimate(
            model_tag=model,
            input_tokens=og_token_count,
            output_tokens=og_output_tokens
        )

        # 5. Calculate Green Metrics Post-Compression
        post_green = eco_estimator.estimate(
            model_tag=model,
            input_tokens=comp_token_count,
            output_tokens=comp_output_tokens
        )

        pre_energy = pre_green.get("energy_wh", {}).get("P50", 0)
        post_energy = post_green.get("energy_wh", {}).get("P50", 0)
        pre_carbon = pre_green.get("carbon_gco2e", {}).get("P50", 0)
        post_carbon = post_green.get("carbon_gco2e", {}).get("P50", 0)

        energy_saved_pct = ((pre_energy - post_energy) / pre_energy * 100) if pre_energy > 0 else 0
        carbon_saved_pct = ((pre_carbon - post_carbon) / pre_carbon * 100) if pre_carbon > 0 else 0

        return jsonify({
            "model_used": model,
            "compression_info": {
                "original_prompt": formatted_prompt,
                "compressed_prompt": compressed_prompt,
                "og_token_count": og_token_count,
                "comp_token_count": comp_token_count,
                "compression_time_ms": round(compression_time, 2)
            },
            "outputs": {
                "pre_compression_response": og_content,
                "post_compression_response": comp_content,
                "pre_output_tokens": og_output_tokens,
                "post_output_tokens": comp_output_tokens
            },
            "green_metrics": {
                "pre_compression": pre_green,
                "post_compression": post_green,
                "savings": {
                    "energy_saved_wh_p50": round(pre_energy - post_energy, 6),
                    "carbon_saved_gco2e_p50": round(pre_carbon - post_carbon, 6),
                    "energy_saved_pct": round(energy_saved_pct, 2),
                    "carbon_saved_pct": round(carbon_saved_pct, 2)
                }
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/green-metric", methods=["POST"])
def compute_green_metric():
    """
    Endpoint 4: Standalone Green Metric Calculator
    """
    data = request.get_json() or {}
    model = data.get("model")
    input_prompt = data.get("input_prompt")
    output_text = data.get("output_text", "")

    if not model or not input_prompt:
        return jsonify({"error": "Fields 'model' and 'input_prompt' are required."}), 400

    try:
        formatted_prompt = PromptFormatter.format_input(input_prompt)
        input_tokens = prompt_compressor.count_tokens(formatted_prompt)
        output_tokens = prompt_compressor.count_tokens(output_text) if output_text else 0

        estimates = eco_estimator.estimate(
            model_tag=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        return jsonify({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "hardware_class": estimates.get("class"),
            "gpu_count": estimates.get("gpu_count"),
            "energy_wh": estimates.get("energy_wh"),
            "carbon_gco2e": estimates.get("carbon_gco2e")
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)