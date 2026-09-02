"""
app.py — ecoGPT Flask server, config-driven.

All model names, endpoints, API keys, and tunables come from
`config.settings`. The four endpoints are unchanged from the contract
documented in README.md.
"""

from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

from config import settings
from router import LLMRouterClient
from prompt_compression import PromptCompressorClient
from green_metric import BlackBoxLLMEstimator
from raw_model_runner import RawModelRunner

PROJECT_ROOT = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app)


class _SafeJSONProvider(DefaultJSONProvider):
    """
    JSON encoder that emits `null` for NaN / +Inf / -Inf instead of the
    non-spec-compliant literal tokens `NaN`, `Infinity`, `-Infinity` that
    Python's stdlib `json` produces by default.

    Why this matters: the green-metric estimator's `reference_rows` come
    straight from the Joule TPS database and contain real `float('nan')`
    values. Without this fix, Flask returns a body that browsers can't
    parse via `response.json()`, which used to silently break the
    rendering of the routing results card.
    """

    @staticmethod
    def default(o):
        # Catch-all for genuinely un-serializable types (custom objects,
        # numpy scalars that don't subclass Python floats, etc.). Emit
        # null instead of raising — the frontend treats null as a missing
        # field, which is safer than a 500.
        return None

    def dumps(self, obj, **kwargs):
        import math as _math
        import json as _json

        kwargs.setdefault("allow_nan", False)
        kwargs.setdefault("default", self.default)

        def _scrub(value):
            if isinstance(value, dict):
                return {k: _scrub(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_scrub(v) for v in value]
            if isinstance(value, float) and (_math.isnan(value) or _math.isinf(value)):
                return None
            return value

        return _json.dumps(_scrub(obj), **kwargs)


# Replace the default provider so float('nan') -> None everywhere.
app.json = _SafeJSONProvider(app)

# Initialize singletons on server start. All config comes from
# `config.settings`; nothing model-specific is hardcoded here.
prompt_compressor = PromptCompressorClient()
router_client = LLMRouterClient()
eco_estimator = BlackBoxLLMEstimator()
raw_runner = RawModelRunner()

# Cache the model configs for ergonomic access throughout the endpoints.
SMALL = settings.models.small
LARGE = settings.models.large
DEFAULT_MODEL = SMALL.name  # Compression defaults to the small model.


def _model_slot_for(model_name: str):
    """
    Return the ModelConfig for `model_name` based on its name match.
    Falls back to None for unknown tags.
    """
    if model_name == SMALL.name:
        return SMALL
    if model_name == LARGE.name:
        return LARGE
    return None


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

    formatted_prompt = PromptFormatter.format_input(raw_prompt)

    try:
        # 1. Route prompt and get primary response
        routed_result = router_client.generate_response(formatted_prompt)
        routed_to = routed_result.get("routed_to", SMALL.name)
        routed_content = PromptFormatter.clean_output(
            routed_result.get("content", "") or ""
        )

        # If the underlying model call silently returned nothing, surface it
        # as a 502 so the frontend renders the red error banner instead of
        # a card full of em-dashes.
        if not routed_content.strip():
            return (
                jsonify(
                    {
                        "error": (
                            "Routed model returned an empty response. "
                            "Check that the configured api_base/api_key are reachable "
                            "and that the model name is valid."
                        )
                    }
                ),
                502,
            )

        # 2. Token counts for routed execution
        input_tokens = prompt_compressor.count_tokens(formatted_prompt)
        routed_output_tokens = prompt_compressor.count_tokens(routed_content)

        # 3. Safe calculation of Green metrics for routed model
        try:
            routed_green = eco_estimator.estimate(
                model_tag=routed_to,
                input_tokens=input_tokens,
                output_tokens=routed_output_tokens,
            )
        except Exception as green_err:
            routed_green = {
                "error": f"Failed to compute green metrics: {str(green_err)}"
            }

        response_payload = {
            "routed_to": routed_to,
            "content": routed_content,
            "input_tokens": input_tokens,
            "output_tokens": routed_output_tokens,
            "green_metrics": routed_green,
            "saved_by_routing": False,
        }

        # 4. If routed to the small model, also run the large one for fidelity.
        if routed_to == SMALL.name:
            try:
                large_raw_content = raw_runner.run_model(
                    model=LARGE.name,
                    prompt=formatted_prompt,
                    api_base=LARGE.api_base,
                    api_key=LARGE.api_key,
                )
                large_content = PromptFormatter.clean_output(large_raw_content)
                large_output_tokens = prompt_compressor.count_tokens(large_content)

                large_green = eco_estimator.estimate(
                    model_tag=LARGE.name,
                    input_tokens=input_tokens,
                    output_tokens=large_output_tokens,
                )

                small_energy = routed_green.get("energy_wh", {}).get("P50", 0)
                large_energy = large_green.get("energy_wh", {}).get("P50", 0)
                small_carbon = routed_green.get("carbon_gco2e", {}).get("P50", 0)
                large_carbon = large_green.get("carbon_gco2e", {}).get("P50", 0)
                small_water = routed_green.get("water_liters", {}).get("P50", 0)
                large_water = large_green.get("water_liters", {}).get("P50", 0)

                energy_saved_pct = (
                    ((large_energy - small_energy) / large_energy * 100)
                    if large_energy > 0
                    else 0
                )
                carbon_saved_pct = (
                    ((large_carbon - small_carbon) / large_carbon * 100)
                    if large_carbon > 0
                    else 0
                )
                water_saved_pct = (
                    ((large_water - small_water) / large_water * 100)
                    if large_water > 0
                    else 0
                )

                response_payload["saved_by_routing"] = True
                response_payload["large_model_comparison"] = {
                    "large_model": LARGE.name,
                    "large_model_content": large_content,
                    "large_model_output_tokens": large_output_tokens,
                    "large_model_green_metrics": large_green,
                    "energy_saved_wh_p50": round(large_energy - small_energy, 6),
                    "carbon_saved_gco2e_p50": round(large_carbon - small_carbon, 6),
                    "water_saved_l_p50": round(large_water - small_water, 6),
                    "energy_saved_pct": round(energy_saved_pct, 2),
                    "carbon_saved_pct": round(carbon_saved_pct, 2),
                    "water_saved_pct": round(water_saved_pct, 2),
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
        slot = _model_slot_for(model)
        api_base = slot.api_base if slot else data.get("api_base")
        api_key = slot.api_key if slot else data.get("api_key")

        if not api_base:
            return (
                jsonify(
                    {
                        "error": (
                            f"No api_base configured for model '{model}'. Either use one "
                            "of the configured models in config.yaml or pass api_base "
                            "in the request body."
                        )
                    }
                ),
                400,
            )

        response_text = raw_runner.run_model(
            model=model,
            prompt=formatted_prompt,
            api_base=api_base,
            api_key=api_key,
        )
        cleaned_response = PromptFormatter.clean_output(response_text)

        return jsonify({"model": model, "content": cleaned_response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compress", methods=["POST"])
def compress_prompt():
    """
    Endpoint 3: Prompt Compressor with Pre/Post Green Metrics
    """
    data = request.get_json() or {}
    prompt = data.get("prompt")
    target_rate = data.get("target_rate")
    model = data.get("model", DEFAULT_MODEL)

    if not prompt:
        return jsonify({"error": "Field 'prompt' is required."}), 400

    # Resolve the model slot up front so we know which api_base/api_key to use.
    slot = _model_slot_for(model)
    if slot is None:
        return (
            jsonify(
                {
                    "error": (
                        f"Model '{model}' is not one of the configured models in "
                        "config.yaml (small / large). Update config.yaml or pass a "
                        "known model name."
                    )
                }
            ),
            400,
        )
    api_base = slot.api_base
    api_key = slot.api_key

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
            og_content = PromptFormatter.clean_output(
                raw_runner.run_model(
                    model=model,
                    prompt=formatted_prompt,
                    api_base=api_base,
                    api_key=api_key,
                )
            )
            og_output_tokens = prompt_compressor.count_tokens(og_content)
        except Exception:
            og_content = ""
            og_output_tokens = 0

        # 3. Run raw model on compressed prompt
        try:
            comp_content = PromptFormatter.clean_output(
                raw_runner.run_model(
                    model=model,
                    prompt=compressed_prompt,
                    api_base=api_base,
                    api_key=api_key,
                )
            )
            comp_output_tokens = prompt_compressor.count_tokens(comp_content)
        except Exception:
            comp_content = ""
            comp_output_tokens = 0

        # 4. Calculate Green Metrics Pre-Compression
        pre_green = eco_estimator.estimate(
            model_tag=model,
            input_tokens=og_token_count,
            output_tokens=og_output_tokens,
        )

        # 5. Calculate Green Metrics Post-Compression
        post_green = eco_estimator.estimate(
            model_tag=model,
            input_tokens=comp_token_count,
            output_tokens=comp_output_tokens,
        )

        pre_energy = pre_green.get("energy_wh", {}).get("P50", 0)
        post_energy = post_green.get("energy_wh", {}).get("P50", 0)
        pre_carbon = pre_green.get("carbon_gco2e", {}).get("P50", 0)
        post_carbon = post_green.get("carbon_gco2e", {}).get("P50", 0)
        pre_water = pre_green.get("water_liters", {}).get("P50", 0)
        post_water = post_green.get("water_liters", {}).get("P50", 0)

        energy_saved_pct = (
            ((pre_energy - post_energy) / pre_energy * 100) if pre_energy > 0 else 0
        )
        carbon_saved_pct = (
            ((pre_carbon - post_carbon) / pre_carbon * 100) if pre_carbon > 0 else 0
        )
        water_saved_pct = (
            ((pre_water - post_water) / pre_water * 100) if pre_water > 0 else 0
        )

        return (
            jsonify(
                {
                    "model_used": model,
                    "compression_info": {
                        "original_prompt": formatted_prompt,
                        "compressed_prompt": compressed_prompt,
                        "og_token_count": og_token_count,
                        "comp_token_count": comp_token_count,
                        "compression_time_ms": round(compression_time, 2),
                    },
                    "outputs": {
                        "pre_compression_response": og_content,
                        "post_compression_response": comp_content,
                        "pre_output_tokens": og_output_tokens,
                        "post_output_tokens": comp_output_tokens,
                    },
                    "green_metrics": {
                        "pre_compression": pre_green,
                        "post_compression": post_green,
                        "savings": {
                            "energy_saved_wh_p50": round(pre_energy - post_energy, 6),
                            "carbon_saved_gco2e_p50": round(
                                pre_carbon - post_carbon, 6
                            ),
                            "water_saved_l_p50": round(pre_water - post_water, 6),
                            "energy_saved_pct": round(energy_saved_pct, 2),
                            "carbon_saved_pct": round(carbon_saved_pct, 2),
                            "water_saved_pct": round(water_saved_pct, 2),
                        },
                    },
                }
            ),
            200,
        )

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
        return (
            jsonify({"error": "Fields 'model' and 'input_prompt' are required."}),
            400,
        )

    try:
        formatted_prompt = PromptFormatter.format_input(input_prompt)
        input_tokens = prompt_compressor.count_tokens(formatted_prompt)
        output_tokens = (
            prompt_compressor.count_tokens(output_text) if output_text else 0
        )

        estimates = eco_estimator.estimate(
            model_tag=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return (
            jsonify(
                {
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "hardware_class": estimates.get("class"),
                    "gpu_count": estimates.get("gpu_count"),
                    "energy_wh": estimates.get("energy_wh"),
                    "carbon_gco2e": estimates.get("carbon_gco2e"),
                    "water_liters": estimates.get("water_liters"),
                    "wue_l_per_kwh": estimates.get("wue_l_per_kwh"),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config", methods=["GET"])
def get_config():
    """
    Expose non-secret runtime config to the frontend so the UI can
    populate model selectors, defaults, etc. without hardcoding.
    """
    return jsonify(
        {
            "models": {
                "small": {"name": SMALL.name, "api_base": SMALL.api_base},
                "large": {"name": LARGE.name, "api_base": LARGE.api_base},
            },
            "router": {
                "type": settings.router.type,
                "threshold": settings.router.threshold,
            },
            "compression": {
                "default_target_rate": settings.compression.target_rate,
            },
        }
    )


@app.route("/", methods=["GET"])
def index():
    """Serve the single-page UI."""
    return send_from_directory(PROJECT_ROOT, "index.html")


@app.route("/index.html", methods=["GET"])
def index_html():
    """Alias for the UI (Flask static-file convention)."""
    return send_from_directory(PROJECT_ROOT, "index.html")


if __name__ == "__main__":
    app.run(
        host=settings.project.host,
        port=settings.project.port,
        debug=True,
        use_reloader=False,
    )
