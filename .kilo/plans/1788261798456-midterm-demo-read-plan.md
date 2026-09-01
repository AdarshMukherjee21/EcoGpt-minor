# Plan: Deeper Look at `midterm_demo` (EcoGpt Router Demo)

## Goal
Produce a deeper, file-by-file read of the `midterm_demo` directory so the user understands exactly what the demo does and how the pieces fit together. This is a research/explanation task, not a code-change task. No source files will be modified.

## What `midterm_demo` Actually Is
Located at `D:\college stuff\sem7\minor ( pinnacle)\EcoGpt -minor\midterm_demo\`. A small Python demo that combines:

1. **LLM routing** via the vendored `RouteLLM/` library (lmsys RouteLLM — a different library than the `LLMRouter/` vendored under `exp_routeLLM/`).
2. **Prompt compression** via Microsoft `llmlingua` (LLMLingua-2).
3. A driver script that measures whether compressing a prompt changes the router's model choice.

It is the "midterm" demo: small, runs end-to-end, used to showcase the EcoGpt cost-saving idea — *smaller / cheaper model is enough for simpler queries, and prompt compression can downgrade "hard" queries to "easy" so the cheap model handles them too.*

## Files To Read (read order)

1. `midterm_demo/run.py` — driver. Already inspected (37 lines).
2. `midterm_demo/router.py` — thin wrapper around `routellm.Controller`. Already inspected (36 lines).
3. `midterm_demo/prompt_compresstion.py` — LLMLingua-2 wrapper with token counting. Already inspected (40 lines).
4. `midterm_demo/question_to_ask.md` — placeholder (0 lines / empty). Note this in the summary.
5. `midterm_demo/RouteLLM/routellm/controller.py` — peek at the `Controller.chat.completions.create` API to explain how `model="router-bert-0.5"` resolves to a `strong_model` or `weak_model`.
6. `midterm_demo/RouteLLM/routellm/routers/routers.py` — confirm available router types (`bert`, `mf`, `causal_llm`, `similarity_weighted`) and how `threshold` is applied.
7. `midterm_demo/RouteLLM/config.example.yaml` — note it as a config template (not yet read).
8. `midterm_demo/RouteLLM/routellm/openai_server.py` — referenced by `router.py` as the server expected to be running on `http://localhost:8000/v1`. Note as a deployment prerequisite.
9. `midterm_demo/RouteLLM/README.md` — short summary of the vendored library to cite.

Stop here — `evals/`, `benchmarks/`, `assets/`, and the `routellm.egg-info/` are not part of the demo's runtime path.

## Decisions Already Made In The Code (read-only, do not change)
- **Router library**: `lmsys/RouteLLM` (vendored as `RouteLLM/`). The `LLMRouter/` under `exp_routeLLM/` is a *different* library (UIUC) and is **not** what the midterm demo uses. Note this distinction in the summary.
- **Router type**: `bert` (default in `run.py`).
- **Threshold**: `0.5` (default).
- **Strong model**: `ollama/gpt-oss:120b-cloud` (local via Ollama).
- **Weak model**: `ollama/qwen2.5:3b` (local via Ollama).
- **Compression model**: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` (CPU).
- **Compression rate**: `0.7` (keep 70% of tokens).
- **OpenAI-compatible endpoint**: `http://localhost:8000/v1` (the `routellm` OpenAI server must be running separately).

## End-to-End Data Flow (what `run.py` demonstrates)
1. Define a finance question (`prompt`).
2. `PromptCompressorClient.count_tokens(prompt)` — `og_token_count`.
3. `PromptCompressorClient.compress_prompt(prompt, target_rate=0.7)` — returns compressed text and latency in ms.
4. Recount tokens after compression — `comp_token_count`.
5. `LLMRouterClient(strong=ollama/gpt-oss:120b-cloud, weak=ollama/qwen2.5:3b).generate_response(prompt)` — calls the local RouteLLM OpenAI server with `model="router-bert-0.5"`. Server picks `strong` or `weak` and forwards. Result includes `routed_to` (which model answered) and `content`.
6. **Re-run** `generate_response` on the *compressed* prompt.
7. Compare `old_model` vs `new_model` — the demo's punchline: compression should often cause the router to pick the cheaper model.

## Open Questions To Ask The User (one at a time, after reading)
1. **What is the demo's success criterion?** Is the goal to (a) show that prompt compression can flip a `big → small` decision on this single example, (b) benchmark cost/quality over the 180-question finance CSV, or (c) just a live runnable demo for the midterm presentation?
2. **Is the empty `question_to_ask.md` intentional** (a placeholder for a demo question) or should the next task populate it?
3. **Is the mid-term target scope** (a) just explain the code, (b) extend it to multi-question benchmarking against the existing routed CSVs, or (c) fix bugs (e.g. `print` is at class body in `router.py:5`, before `__init__` is defined — harmless but ugly)?

## Risks / Issues Spotted In Existing Code (informational only)
- `midterm_demo/router.py:5` — `print("Initializing LLMRouterClient...")` is at class scope, runs once at *import time*, not at instantiation. Harmless but misleading.
- `midterm_demo/router.py:7-9` — `OPENAI_API_KEY` is hard-coded to a dummy string and `OPENAI_API_BASE` is hard-coded to `localhost:8000`. Side effects on import make this module fragile if imported multiple times.
- `midterm_demo/run.py:11-15` — prints `comp_token_count` and then re-prints `compressed_prompt` text, but never prints the *compressed* prompt's actual content, which makes output noisy. Read-only observation.
- `midterm_demo/run.py:32-37` — the "new model" comparison uses the *compressed prompt* on the second router call, but the first call's response is never used; the demo doesn't actually compare response quality, only model selection.
- `midterm_demo/question_to_ask.md` is empty (0 lines).
- The demo silently depends on a locally-running `routellm` OpenAI server (`openai_server.py`) and on Ollama having both `gpt-oss:120b-cloud` and `qwen2.5:3b` pulled.

## Validation / Out-of-Scope
- No tests to run; this is research code.
- No CI / lint configured at this directory level.
- Out of scope: changing routing strategy, replacing LLMLingua with another compressor, porting the demo to use the `LLMRouter/` library instead of `RouteLLM/`.

## Next Step After Plan Exit
Hand the saved plan back to the user. Expected follow-up: pick a question from the "Open Questions" section, then either (a) switch to an implementation-capable agent to make any requested code changes, or (b) keep this agent in plan/research mode to deepen the read (e.g. open `controller.py` and `routers.py` to document how `router-bert-0.5` is actually scored).
