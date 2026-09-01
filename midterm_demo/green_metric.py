import re
import numpy as np
import pandas as pd


class BlackBoxLLMEstimator:
    """
    BLACK-BOX LLM ENVIRONMENTAL IMPACT ESTIMATOR

    PIPELINE
    Black-box model tag
          ↓
    HHAI-inspired model size classification
          ↓
    Assumed GPU count
          ↓
    Find comparable models from Joule TPS database
          ↓
    Representative aggregate production TPS
          ↓
    Joule-style Monte Carlo energy estimate
          ↓
    Carbon = Energy × Carbon Intensity

    IMPORTANT:
    This is a COLLEGE-PROJECT ESTIMATOR.
    It produces scenario-based estimates.
    It does NOT measure the actual infrastructure used by
    OpenAI, Anthropic, Google, etc.
    """

    # Default Constants
    DEFAULT_N_SAMPLES = 10_000
    DEFAULT_RANDOM_SEED = 42

    JOULE_CSV_URL = (
        "https://raw.githubusercontent.com/"
        "microsoft/aienergy_simulation/main/"
        "model_throughput_DB.csv"
    )

    # HHAI-inspired parameter-size hardware scaling
    CLASS_GPU_COUNT = {
        "Nano": 1,
        "Micro": 1,
        "Small": 2,
        "Medium": 4,
        "Large": 8
    }

    # External reference models for classes missing from Joule
    EXTERNAL_REFERENCES = {
        # NANO: < 7B
        "Nano": {
            "reference_model": "Llama 3.2 3B",
            "model_type": "Dense",
            "hardware": "1x H100",
            "gpu_count": 1,
            "tps": 18_000,
            "source": "External H100 benchmark",
            "assumption": (
                "Llama 3.2 3B used as representative "
                "for <7B black-box models."
            )
        },
        # MICRO: 7B - 20B
        "Micro": {
            "reference_model": "Llama 3.1 8B",
            "model_type": "Dense",
            "hardware": "1x H100",
            "gpu_count": 1,
            "tps": 15_270,
            "source": "NVIDIA H100 inference benchmark",
            "assumption": (
                "Llama 3.1 8B used as representative "
                "for 7B-20B black-box models."
            )
        },
        # SMALL: 20B - 40B
        "Small": {
            "reference_model": "20B-40B reference model",
            "model_type": "Approximate",
            "hardware": "2x H100",
            "gpu_count": 2,
            "tps": 10_000,
            "source": "Project approximation",
            "assumption": (
                "Approximate throughput used because a clean "
                "matched dense 20B-40B H100 benchmark was not "
                "available in our selected benchmark sources."
            )
        }
    }

    # H100 Hardware Power Assumptions
    H100_8GPU_MAX_POWER_KW = 10.2

    # Monte Carlo Uncertainty Assumptions
    POWER_P5_FRACTION = 0.40
    POWER_P95_FRACTION = 0.90
    PUE_P5 = 1.05
    PUE_P95 = 1.40

    # Grid Carbon Intensity Assumption (kg CO2e / kWh or g CO2e / Wh)
    CARBON_INTENSITY_KG_PER_KWH = 0.3528

    def __init__(
        self,
        joule_source=None,
        n_samples=DEFAULT_N_SAMPLES,
        random_seed=DEFAULT_RANDOM_SEED
    ):
        """
        Initialize the estimator and prepare the Joule TPS database.
        """
        self.n_samples = n_samples
        self.random_seed = random_seed

        if joule_source is None:
            joule_source = self.JOULE_CSV_URL

        if isinstance(joule_source, pd.DataFrame):
            self.joule_df = joule_source.copy()
        else:
            self.joule_df = pd.read_csv(joule_source)

        self._prepare_joule_df()

    def _prepare_joule_df(self):
        """
        Prepare and clean the internal Joule TPS database DataFrame.
        """
        # Rename columns to simpler internal names
        self.joule_df = self.joule_df.rename(columns={
            "Total Parameter Count (B)": "Total Parameter Count",
            "Active Parameter Count (B)": "Active Parameter Count",
            "Type (Dense / MoE)": "Model Type",
            "Tokens per Second (TPS)": "TPS",
            "Power for Node (kW)": "Node Power kW",
            "Confidence level": "Confidence"
        })

        # Convert required numerical columns to numbers
        numeric_columns = [
            "Total Parameter Count",
            "Active Parameter Count",
            "TPS",
            "Input Length",
            "Output Length",
            "Batch Size",
            "Node Power kW"
        ]

        for column in numeric_columns:
            if column in self.joule_df.columns:
                self.joule_df[column] = pd.to_numeric(
                    self.joule_df[column],
                    errors="coerce"
                )

        # Classify each Joule model using HHAI size classes
        self.joule_df["HHAI_Class"] = (
            self.joule_df["Total Parameter Count"]
            .apply(self._joule_classify)
        )

    def _joule_classify(self, params):
        if pd.isna(params):
            return None
        return self.classify_parameter_count(float(params))

    @staticmethod
    def classify_parameter_count(parameters_billion):
        """
        HHAI parameter-size bands:
        Nano   : < 7B
        Micro  : 7B - 20B
        Small  : 20B - 40B
        Medium : 40B - 70B
        Large  : > 70B
        """
        if parameters_billion < 7:
            return "Nano"
        elif parameters_billion < 20:
            return "Micro"
        elif parameters_billion < 40:
            return "Small"
        elif parameters_billion <= 70:
            return "Medium"
        else:
            return "Large"

    @staticmethod
    def extract_parameter_count(model_tag):
        """
        Try to obtain a parameter count directly from a model name.

        Examples:
            llama-3.1-8b   -> 8
            qwen-32b       -> 32
            model-405b     -> 405

        Returns None when the model tag does not expose a parameter count.
        """
        model_tag = model_tag.lower()
        matches = re.findall(
            r"(\d+(?:\.\d+)?)\s*[bB]\b",
            model_tag
        )
        if not matches:
            return None
        return float(matches[-1])

    def classify_blackbox_model(self, model_tag):
        """
        Determine an HHAI-inspired hardware class from a model tag.

        Priority:
        1. If parameter count appears in the tag:
               use the HHAI parameter bands.
        2. If parameters are hidden:
               "nano"  -> Small
               "mini"  -> Medium
               other proprietary/flagship -> Large
        """
        tag = model_tag.lower()

        # STEP 1: Look for explicit parameter count
        parameters_b = self.extract_parameter_count(tag)

        if parameters_b is not None:
            model_class = self.classify_parameter_count(parameters_b)
            classification_method = (
                f"Parameter count found in model tag ({parameters_b}B)"
            )
        # STEP 2: HHAI-style naming heuristic
        elif "nano" in tag:
            # HHAI classified GPT-4.1 nano as Small rather than assuming <7B.
            model_class = "Small"
            classification_method = (
                "Undisclosed parameters; HHAI-style Nano naming heuristic"
            )
        elif "mini" in tag:
            model_class = "Medium"
            classification_method = (
                "Undisclosed parameters; HHAI-style Mini naming heuristic"
            )
        elif "tiny" in tag or "micro" in tag:
            model_class = "Micro"
            classification_method = (
                "Undisclosed parameters; project naming heuristic"
            )
        else:
            model_class = "Large"
            classification_method = (
                "Undisclosed parameters; assumed flagship/Large"
            )

        gpu_count = self.CLASS_GPU_COUNT[model_class]

        return {
            "model_tag": model_tag,
            "parameters_b": parameters_b,
            "class": model_class,
            "gpu_count": gpu_count,
            "classification_method": classification_method
        }

    def get_reference_tps(
        self,
        model_class,
        input_tokens,
        output_tokens,
        top_k=5
    ):
        """
        Estimate aggregate production TPS for a black-box model.

        METHOD:
        1. Filter Joule database to models in the same HHAI class.
        2. Prefer H100 benchmark records.
        3. Find records with input/output token lengths closest to the query.
        4. Use the median TPS of the nearest records.
        """
        subset = self.joule_df[
            self.joule_df["HHAI_Class"] == model_class
        ].copy()

        # Prefer H100 benchmark records
        h100_subset = subset[
            subset["Hardware"]
            .astype(str)
            .str.contains("H100", case=False, na=False)
        ]

        if len(h100_subset) > 0:
            subset = h100_subset

        if len(subset) == 0:
            raise ValueError(
                f"No Joule TPS records found for class {model_class}"
            )

        # Make sure columns are numeric
        for column in ["TPS", "Input Length", "Output Length"]:
            subset[column] = pd.to_numeric(subset[column], errors="coerce")

        subset = subset.dropna(
            subset=["TPS", "Input Length", "Output Length"]
        )

        # QUERY-SIZE DISTANCE using log distance
        subset["distance"] = (
            np.abs(
                np.log1p(subset["Input Length"]) - np.log1p(input_tokens)
            )
            + np.abs(
                np.log1p(subset["Output Length"]) - np.log1p(output_tokens)
            )
        )

        nearest = subset.sort_values("distance").head(top_k)
        reference_tps = nearest["TPS"].median()

        return float(reference_tps), nearest

    def get_blackbox_reference(
        self,
        model_class,
        input_tokens,
        output_tokens
    ):
        """
        Select throughput reference for a black-box model.

        Nano / Micro / Small: Use external reference table.
        Medium / Large: Use the Joule TPS database.
        """
        if model_class in self.EXTERNAL_REFERENCES:
            ref = self.EXTERNAL_REFERENCES[model_class]
            return {
                "reference_model": ref["reference_model"],
                "hardware": ref["hardware"],
                "gpu_count": ref["gpu_count"],
                "tps": ref["tps"],
                "source": ref["source"],
                "assumption": ref["assumption"],
                "reference_rows": None
            }

        reference_tps, rows = self.get_reference_tps(
            model_class=model_class,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        return {
            "reference_model": "Joule same-class models",
            "hardware": "H100 reference",
            "gpu_count": self.CLASS_GPU_COUNT[model_class],
            "tps": reference_tps,
            "source": "Oviedo et al. Joule TPS database",
            "assumption": (
                "Median TPS of nearest benchmark workloads "
                "from models in the same HHAI size class."
            ),
            "reference_rows": rows
        }

    def estimate_max_power(self, gpu_count):
        """
        Scale max power from the 8-GPU H100 node power.
        """
        return self.H100_8GPU_MAX_POWER_KW * gpu_count / 8

    @staticmethod
    def sample_lognormal_from_p5_p95(p5, p95, n_samples, rng):
        """
        Generate lognormal distribution samples from P5 and P95 percentiles.
        """
        z = 1.6448536269514722
        mu = (np.log(p5) + np.log(p95)) / 2
        sigma = (np.log(p95) - np.log(p5)) / (2 * z)
        return rng.lognormal(mean=mu, sigma=sigma, size=n_samples)

    def estimate_energy(
        self,
        output_tokens,
        production_tps,
        max_power_kw,
        n_samples=None,
        seed=None
    ):
        """
        Joule-style Monte Carlo energy estimate.

        E = PUE × Power × EffectiveTokens / TPS
        """
        if n_samples is None:
            n_samples = self.n_samples
        if seed is None:
            seed = self.random_seed

        rng = np.random.default_rng(seed)

        # Power uncertainty
        power_samples = self.sample_lognormal_from_p5_p95(
            p5=max_power_kw * self.POWER_P5_FRACTION,
            p95=max_power_kw * self.POWER_P95_FRACTION,
            n_samples=n_samples,
            rng=rng
        )

        # PUE uncertainty
        pue_samples = self.sample_lognormal_from_p5_p95(
            p5=self.PUE_P5,
            p95=self.PUE_P95,
            n_samples=n_samples,
            rng=rng
        )

        attributable_seconds = output_tokens / production_tps

        energy_wh = (
            power_samples
            * attributable_seconds
            * pue_samples
            * 1000
            / 3600
        )

        return energy_wh

    def energy_to_carbon(
        self,
        energy_wh,
        carbon_intensity=None
    ):
        """
        Convert energy in Wh to carbon emissions in g CO2e.
        """
        if carbon_intensity is None:
            carbon_intensity = self.CARBON_INTENSITY_KG_PER_KWH

        return energy_wh * carbon_intensity

    @staticmethod
    def summarize_distribution(samples):
        """
        Calculate percentiles P5, P25, P50, P75, P95 for a sample set.
        """
        return {
            "P5": np.percentile(samples, 5),
            "P25": np.percentile(samples, 25),
            "P50": np.percentile(samples, 50),
            "P75": np.percentile(samples, 75),
            "P95": np.percentile(samples, 95)
        }

    def estimate(
        self,
        model_tag,
        input_tokens,
        output_tokens,
        carbon_intensity=None
    ):
        """
        Run the full estimation pipeline for a black-box model request.
        """
        if carbon_intensity is None:
            carbon_intensity = self.CARBON_INTENSITY_KG_PER_KWH

        # STEP 1: Model classification
        model_info = self.classify_blackbox_model(model_tag)
        model_class = model_info["class"]
        gpu_count = model_info["gpu_count"]

        # STEP 2: Estimate hardware power
        max_power_kw = self.estimate_max_power(gpu_count)

        # STEP 3: Select reference TPS
        reference = self.get_blackbox_reference(
            model_class=model_class,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        reference_tps = reference["tps"]

        # STEP 4: Monte Carlo energy estimation
        energy_samples = self.estimate_energy(
            output_tokens=output_tokens,
            production_tps=reference_tps,
            max_power_kw=max_power_kw
        )
        energy_summary = self.summarize_distribution(energy_samples)

        # STEP 5: Carbon estimation
        carbon_samples = self.energy_to_carbon(
            energy_samples,
            carbon_intensity
        )
        carbon_summary = self.summarize_distribution(carbon_samples)

        return {
            "model": model_tag,
            "class": model_class,
            "classification_method": model_info["classification_method"],
            "gpu_count": gpu_count,
            "max_power_kw": max_power_kw,
            "reference_model": reference["reference_model"],
            "reference_tps": reference_tps,
            "reference_source": reference["source"],
            "reference_assumption": reference["assumption"],
            "energy_wh": energy_summary,
            "carbon_gco2e": carbon_summary,
            "reference_rows": reference["reference_rows"]
        }


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == "__main__":
    estimator = BlackBoxLLMEstimator()

    result = estimator.estimate(
        model_tag="llama-3.1-8b",
        input_tokens=500,
        output_tokens=300
    )

    print("BLACK-BOX MODEL ESTIMATE")
    print("========================")
    print("Model:", result["model"])
    print("HHAI Class:", result["class"])
    print("Classification:", result["classification_method"])

    print("\nHARDWARE ASSUMPTION")
    print("-------------------")
    print("GPU Count:", result["gpu_count"])
    print(f"Max Power: {result['max_power_kw']:.2f} kW")

    print("\nTPS REFERENCE")
    print("-------------")
    print("Reference Model:", result["reference_model"])
    print(f"Reference TPS: {result['reference_tps']:,.2f}")
    print("Source:", result["reference_source"])
    print("Assumption:", result["reference_assumption"])

    print("\nENERGY")
    print("------")
    for p, value in result["energy_wh"].items():
        print(f"{p}: {value:.4f} Wh")

    print("\nCARBON")
    print("------")
    for p, value in result["carbon_gco2e"].items():
        print(f"{p}: {value:.4f} gCO2e")