import abc
import functools
import os
import random

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from routellm.routers.causal_llm.configs import RouterModelConfig
from routellm.routers.causal_llm.llm_utils import (
    load_prompt_format,
    to_openai_api_messages,
)
from routellm.routers.causal_llm.model import CausalLLMClassifier
from routellm.routers.matrix_factorization.model import MODEL_IDS, MFModel
from routellm.routers.similarity_weighted.utils import (
    OPENAI_CLIENT,
    compute_elo_mle_with_tie,
    compute_tiers,
    preprocess_battles,
)


def no_parallel(cls):
    cls.NO_PARALLEL = True

    return cls


class Router(abc.ABC):
    NO_PARALLEL = False

    # Returns a float between 0 and 1 representing the value used to route to models, conventionally the winrate of the strong model.
    # If this value is >= the user defined cutoff, the router will route to the strong model, otherwise, it will route to the weak model.
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        pass

    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak

    def __str__(self):
        return NAME_TO_CLS[self.__class__]


@no_parallel
class CausalLLMRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        score_threshold=4,
        special_tokens=["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"],
        num_outputs=5,
        model_type="causal",
        model_id="meta-llama/Meta-Llama-3-8B",
        flash_attention_2=False,
    ):
        model_config = RouterModelConfig(
            model_id=model_id,
            model_type=model_type,
            flash_attention_2=flash_attention_2,
            special_tokens=special_tokens,
            num_outputs=num_outputs,
        )
        prompt_format = load_prompt_format(model_config.model_id)
        self.router_model = CausalLLMClassifier(
            config=model_config,
            ckpt_local_path=checkpoint_path,
            score_threshold=score_threshold,
            prompt_format=prompt_format,
            prompt_field="messages",
            additional_fields=[],
            use_last_turn=True,
        )
        system_message = hf_hub_download(
            repo_id=checkpoint_path, filename="system_ft_v5.txt"
        )
        classifier_message = hf_hub_download(
            repo_id=checkpoint_path, filename="classifier_ft_v5.txt"
        )
        with open(system_message) as pr:
            system_message = pr.read()
        with open(classifier_message) as pr:
            classifier_message = pr.read()
        self.to_openai_messages = functools.partial(
            to_openai_api_messages, system_message, classifier_message
        )

    def calculate_strong_win_rate(self, prompt):
        input = {}
        input["messages"] = self.to_openai_messages([prompt])
        output = self.router_model(input)
        if output is None:
            # Route to strong model if output is invalid
            return 1
        else:
            return 1 - output["binary_prob"]


@no_parallel
class BERTRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        num_labels=3,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    def calculate_strong_win_rate(self, prompt):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.numpy()[0]

        exp_scores = np.exp(logits - np.max(logits))
        softmax_scores = exp_scores / np.sum(exp_scores)

        # Compute prob of label 1 and 2 (tie, tier 2 wins)
        binary_prob = np.sum(softmax_scores[-2:])
        return 1 - binary_prob


class SWRankingRouter(Router):
    def __init__(
        self,
        arena_battle_datasets,
        arena_embedding_datasets,
        # This is the model pair for Elo calculations at inference time,
        # and can be different from the model pair used for routing.
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        num_tiers=10,
    ):
        self.strong_model = strong_model
        self.weak_model = weak_model

        self.arena_df = concatenate_datasets(
            [load_dataset(dataset, split="train") for dataset in arena_battle_datasets]
        ).to_pandas()
        self.arena_df = preprocess_battles(self.arena_df)

        embeddings = [
            np.array(load_dataset(dataset, split="train").to_dict()["embeddings"])
            for dataset in arena_embedding_datasets
        ]
        self.arena_conv_embedding = np.concatenate(embeddings)
        self.embedding_model = "text-embedding-3-small"

        assert len(self.arena_df) == len(self.arena_conv_embedding), (
            "Number of battle embeddings is mismatched to data"
        )

        model_ratings = compute_elo_mle_with_tie(self.arena_df)
        self.model2tier = compute_tiers(model_ratings, num_tiers=num_tiers)

        self.arena_df["model_a"] = self.arena_df["model_a"].apply(
            lambda x: self.model2tier[x]
        )
        self.arena_df["model_b"] = self.arena_df["model_b"].apply(
            lambda x: self.model2tier[x]
        )

    def get_weightings(self, similarities):
        max_sim = np.max(similarities)
        return 10 * 10 ** (similarities / max_sim)

    def calculate_strong_win_rate(
        self,
        prompt,
    ):
        prompt_emb = (
            (
                OPENAI_CLIENT.embeddings.create(
                    input=[prompt], model=self.embedding_model
                )
            )
            .data[0]
            .embedding
        )
        similarities = np.dot(self.arena_conv_embedding, prompt_emb) / (
            np.linalg.norm(self.arena_conv_embedding, axis=1)
            * np.linalg.norm(prompt_emb)
        )

        weightings = self.get_weightings(similarities)
        res = compute_elo_mle_with_tie(self.arena_df, sample_weight=weightings)

        weak_score, strong_score = (
            res[self.model2tier[self.weak_model]],
            res[self.model2tier[self.strong_model]],
        )
        weak_winrate = 1 / (1 + 10 ** ((strong_score - weak_score) / 400))
        strong_winrate = 1 - weak_winrate

        # If the expected strong winrate is greater than the threshold, use strong
        return strong_winrate


def _find_local_checkpoint(checkpoint_path):
    """
    If `checkpoint_path` points at a local directory (the
    PyTorchModelHubMixin layout) or a local weights file, return the path
    to load; otherwise return None (meaning "treat as a HF hub repo id").
    """
    if not checkpoint_path:
        return None
    expanded = os.path.expanduser(checkpoint_path)
    if os.path.isdir(expanded):
        for name in ("pytorch_model.bin", "model.pt", "model.bin"):
            candidate = os.path.join(expanded, name)
            if os.path.isfile(candidate):
                return candidate
        return None
    if os.path.isfile(expanded):
        return expanded
    return None


@no_parallel
class MatrixFactorizationRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        # This is the model pair for scoring at inference time,
        # and can be different from the model pair used for routing.
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        hidden_size=128,
        num_models=64,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
        # "openai" keeps the stock behavior (OPENAI_CLIENT, real API or
        # host-app monkey-patch); "sentence-transformers" embeds prompts
        # locally with `embedding_model` — no OpenAI key needed.
        embedding_provider="openai",
        embedding_model=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        local_checkpoint = _find_local_checkpoint(checkpoint_path)
        if local_checkpoint is not None:
            # A local checkpoint (e.g. a MiniLM-trained `mf` from the Colab
            # notebook). Derive the architecture from the weights so the
            # caller doesn't need to know text_dim / dim / num_models.
            state = torch.load(local_checkpoint, map_location="cpu")
            projection_key = (
                "text_proj.0.weight"
                if "text_proj.0.weight" in state
                else "text_proj.weight"
            )
            classifier_key = (
                "classifier.0.weight"
                if "classifier.0.weight" in state
                else "classifier.weight"
            )
            hidden_size = state[projection_key].shape[0]
            text_dim = state[projection_key].shape[1]
            num_models = state["P.weight"].shape[0]
            num_classes = state[classifier_key].shape[0]
            use_proj = True

            self.model = MFModel(
                dim=hidden_size,
                num_models=num_models,
                text_dim=text_dim,
                num_classes=num_classes,
                use_proj=use_proj,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )
            self.model.load_state_dict(state)
            print(
                f"[mf] loaded local checkpoint {local_checkpoint} "
                f"(dim={hidden_size}, text_dim={text_dim}, "
                f"num_models={num_models}, num_classes={num_classes})"
            )
        else:
            self.model = MFModel.from_pretrained(
                checkpoint_path,
                dim=hidden_size,
                num_models=num_models,
                text_dim=text_dim,
                num_classes=num_classes,
                use_proj=use_proj,
            )
            self.model.embedding_provider = embedding_provider
            if embedding_model:
                self.model.embedding_model = embedding_model
            print(f"[mf] loaded hub checkpoint {checkpoint_path}")

        self.model = self.model.eval().to(device)
        try:
            self.strong_model_id = MODEL_IDS[strong_model]
            self.weak_model_id = MODEL_IDS[weak_model]
        except KeyError:
            # The scoring pair must be one of the 64 arena models. If the
            # host app passes model names that are not in MODEL_IDS, fall
            # back to the default scoring pair so routing still works.
            print(
                f"[mf] WARNING: scoring pair ({strong_model}, {weak_model}) not "
                "in MODEL_IDS, falling back to (gpt-4-1106-preview, "
                "mixtral-8x7b-instruct-v0.1)."
            )
            self.strong_model_id = MODEL_IDS["gpt-4-1106-preview"]
            self.weak_model_id = MODEL_IDS["mixtral-8x7b-instruct-v0.1"]

    def calculate_strong_win_rate(self, prompt):
        winrate = self.model.pred_win_rate(
            self.strong_model_id, self.weak_model_id, prompt
        )
        return winrate


# Parallelism makes the randomness non deterministic
@no_parallel
class RandomRouter(Router):
    def calculate_strong_win_rate(
        self,
        prompt,
    ):
        del prompt
        return random.uniform(0, 1)


ROUTER_CLS = {
    "random": RandomRouter,
    "mf": MatrixFactorizationRouter,
    "causal_llm": CausalLLMRouter,
    "bert": BERTRouter,
    "sw_ranking": SWRankingRouter,
}
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
