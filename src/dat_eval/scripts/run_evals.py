"""Run DAT, CDAT, and PACE evaluations against LLMs via OpenRouter.

Step 1 of the pipeline: queries each model for all three tasks and saves
raw responses.

This makes paid API calls. The run is resumable: a (model, test, temperature) whose response
file already exists in output_dir is skipped, so an interrupted run can be restarted as is.

Usage:
    python -m src.dat_eval.scripts.run_evals configs/run_evals.yaml [--debug]
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from src.utils import load_config, save_config
from src.dat_eval.llm import (
    call_llm_async,
    extract_words_from_response,
    get_async_client,
    model_id_to_key,
)
from src.dat_eval.dat import DAT_PROMPT
from src.dat_eval.cdat import cdat_prompt, DEFAULT_CUES
from src.dat_eval.pace import (
    pace_stage1_prompt,
    pace_stage2_prompt,
    DEFAULT_SEEDS,
)

# OpenRouter prices in USD per million (input, output) tokens, used only for the optional
# budget cap. Fill them in from https://openrouter.ai/models in the config under `pricing:`
# ({model_id: [in, out]}). With a budget cap set, a model without a price counts as infinitely
# expensive, so the cap stops before it.
PRICING: dict[str, tuple[float, float]] = {}

# Per-call token estimates (based on observed responses)
CALL_COST = {
    "dat":   (150, 400),
    "cdat":  (200, 400),
    "pace1": (200, 300),
    "pace2": (300, 600),
}

# Models that generate reasoning tokens internally. For these we need a larger
# max_tokens budget because reasoning tokens count toward the cap (even when
# `reasoning.exclude=true` is set — they are generated, just not returned).
REASONING_MODELS = {
    "qwen/qwq-32b",
    "deepseek/deepseek-r1",
    "openai/o3",
    "openai/o3-mini",
    "openai/o4-mini",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    # Claude Sonnet 4.6 emits extended-thinking tokens by default that burn
    # through a 256-token cap before any visible content is produced.
    "anthropic/claude-sonnet-4.6",
    # Gemini 3.x models think-by-default and ignore reasoning.exclude; truncate
    # JSON mid-array under the base 256-token cap. Add to the reasoning bucket
    # so they get 4x the max_tokens.
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    # OpenAI GPT-5.2 family similarly emits reasoning by default.
    "openai/gpt-5.2",
    "openai/gpt-5.2-pro",
}


def max_tokens_for(model_id: str, base: int, reasoning_multiplier: int = 4) -> int:
    """Return max_tokens for a model. Reasoning models get a larger budget.

    Non-reasoning models: `base` (enough for the expected output, no more — this
    prevents runaway token generation at high temperature).
    Reasoning models: `base * reasoning_multiplier` to account for internal
    reasoning tokens.
    """
    if model_id in REASONING_MODELS:
        return base * reasoning_multiplier
    return base


def estimate_model_cost(
    model_id: str,
    pending_dat_temps: int,
    pending_cdat_temps: int,
    pending_pace_temps: int,
    dat_trials: int,
    n_cues: int,
    n_seeds: int,
) -> float:
    """Estimate USD cost of running remaining work for a given model."""
    # Local serving (LLM_BASE_URL set) is free; the OpenRouter price
    # table does not apply. Behavior is unchanged when serving via
    # OpenRouter (env unset).
    if os.environ.get("LLM_BASE_URL"):
        return 0.0
    pricing = PRICING.get(model_id)
    if pricing is None:
        return float("inf")  # can't run what we can't price
    in_price, out_price = pricing

    total_in, total_out = 0, 0
    if pending_dat_temps > 0:
        n_calls = pending_dat_temps * dat_trials
        total_in += n_calls * CALL_COST["dat"][0]
        total_out += n_calls * CALL_COST["dat"][1]
    if pending_cdat_temps > 0:
        n_calls = pending_cdat_temps * n_cues
        total_in += n_calls * CALL_COST["cdat"][0]
        total_out += n_calls * CALL_COST["cdat"][1]
    if pending_pace_temps > 0:
        total_in += pending_pace_temps * n_seeds * CALL_COST["pace1"][0]
        total_out += pending_pace_temps * n_seeds * CALL_COST["pace1"][1]
        total_in += pending_pace_temps * n_seeds * 3 * CALL_COST["pace2"][0]
        total_out += pending_pace_temps * n_seeds * 3 * CALL_COST["pace2"][1]

    return (total_in * in_price + total_out * out_price) / 1_000_000


async def _run_one(async_client, sem, call_kwargs) -> tuple[str | None, Exception | None]:
    """Make a single LLM call with a shared semaphore. Returns (text, error).

    Treats a None response content as an error (some providers return null
    content when they refuse / content-filter / fail silently).
    """
    async with sem:
        try:
            raw = await call_llm_async(async_client, **call_kwargs)
            if raw is None:
                return None, RuntimeError("provider returned null content")
            return raw, None
        except Exception as e:
            return None, e


async def run_dat(
    async_client,
    sem,
    model_id: str,
    n_trials: int,
    temperature: float,
    base_seed: int = 1000,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int = 256,
    reasoning: dict | None = None,
) -> list[dict]:
    """Run DAT evaluation concurrently: generate n_trials sets of 10 divergent words."""
    async def one_trial(trial):
        seed = base_seed + trial
        raw, err = await _run_one(async_client, sem, dict(
            messages=[{"role": "user", "content": DAT_PROMPT}],
            model=model_id,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            reasoning=reasoning,
        ))
        if err is not None:
            return {"trial": trial, "seed": seed, "raw_response": None,
                    "words": [], "api_error": f"{type(err).__name__}: {err}"}
        words = extract_words_from_response(raw, expected_count=10)
        return {"trial": trial, "seed": seed, "raw_response": raw,
                "words": words, "api_error": None}

    return await asyncio.gather(*[one_trial(i) for i in range(n_trials)])


async def run_cdat(
    async_client,
    sem,
    model_id: str,
    cues: list[str],
    temperature: float,
    base_seed: int = 2000,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int = 256,
    reasoning: dict | None = None,
) -> dict[str, dict]:
    """Run CDAT concurrently: 10 associated-but-diverse words per cue."""
    async def one_cue(i, cue):
        seed = base_seed + i
        raw, err = await _run_one(async_client, sem, dict(
            messages=[{"role": "user", "content": cdat_prompt(cue)}],
            model=model_id,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            reasoning=reasoning,
        ))
        if err is not None:
            return cue, {"seed": seed, "raw_response": None,
                         "words": [], "api_error": f"{type(err).__name__}: {err}"}
        words = extract_words_from_response(raw, expected_count=10)
        return cue, {"seed": seed, "raw_response": raw,
                     "words": words, "api_error": None}

    pairs = await asyncio.gather(*[one_cue(i, c) for i, c in enumerate(cues)])
    return dict(pairs)


async def run_pace(
    async_client,
    sem,
    model_id: str,
    seeds: list[str],
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
    stage1_max_tokens: int = 400,
    stage2_max_tokens: int = 1200,
    reasoning: dict | None = None,
) -> dict[str, dict]:
    """Run PACE concurrently: 3 association chains of 20 words per seed.

    Stage 1 (seed -> 3 first-associations) runs across all seeds concurrently.
    Stage 2 (each first-association -> 20-word chain) runs concurrently across
    all (seed, chain) pairs.
    """
    # Stage 1: one call per seed, all concurrent
    async def stage1_for_seed(seed_word):
        raw, err = await _run_one(async_client, sem, dict(
            messages=[{"role": "user", "content": pace_stage1_prompt(seed_word)}],
            model=model_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=stage1_max_tokens,
            reasoning=reasoning,
        ))
        if err is not None:
            return seed_word, {"stage1_raw": None, "stage1_parsed": [],
                               "stage1_error": f"{type(err).__name__}: {err}"}
        parsed = _parse_pace_stage1(raw)
        return seed_word, {"stage1_raw": raw, "stage1_parsed": parsed, "stage1_error": None}

    stage1_pairs = await asyncio.gather(*[stage1_for_seed(s) for s in seeds])
    stage1 = dict(stage1_pairs)

    # Stage 2: one call per (seed, first-association), all concurrent
    async def stage2_for_assoc(seed_word, assoc):
        raw, err = await _run_one(async_client, sem, dict(
            messages=[{"role": "user", "content": pace_stage2_prompt(
                seed_word, assoc["word"], assoc.get("reason", ""))}],
            model=model_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=stage2_max_tokens,
            reasoning=reasoning,
        ))
        if err is not None:
            return {"first_association": assoc, "raw_response": None,
                    "chain": [seed_word, assoc["word"]],
                    "api_error": f"{type(err).__name__}: {err}"}
        chain = _parse_pace_stage2(raw, seed_word, assoc["word"])
        return {"first_association": assoc, "raw_response": raw,
                "chain": chain, "api_error": None}

    # Build the list of all (seed, assoc) pairs we need to run
    stage2_tasks = []
    stage2_keys = []
    for seed_word in seeds:
        s1 = stage1[seed_word]
        for assoc in s1["stage1_parsed"]:
            stage2_tasks.append(stage2_for_assoc(seed_word, assoc))
            stage2_keys.append(seed_word)

    stage2_results = await asyncio.gather(*stage2_tasks)

    # Bucket stage-2 results back by seed
    results = {seed_word: {"stage1_raw": stage1[seed_word]["stage1_raw"],
                           "stage1_parsed": stage1[seed_word]["stage1_parsed"],
                           "chains": []} for seed_word in seeds}
    for seed_word, chain_result in zip(stage2_keys, stage2_results):
        results[seed_word]["chains"].append(chain_result)

    return results


def _parse_pace_stage1(raw: str) -> list[dict]:
    """Parse PACE stage 1 response into list of {word, reason} dicts."""
    import re

    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "results" in data and isinstance(data["results"], list):
                return [
                    {"word": r.get("word", ""), "reason": r.get("reason", "")}
                    for r in data["results"][:3]
                ]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: extract words from response
    words = extract_words_from_response(raw, expected_count=3)
    return [{"word": w, "reason": ""} for w in words[:3]]


def _parse_pace_stage2(raw: str, seed: str, second_word: str) -> list[str]:
    """Parse PACE stage 2 response into a word chain."""
    import re

    chain = [seed, second_word]

    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "results" in data and isinstance(data["results"], list):
                for entry in data["results"]:
                    word = entry.get("word", "").strip().lower()
                    if word and word not in chain:
                        chain.append(word)
                    elif word:
                        chain.append(word)  # Allow duplicates in chain
                return chain[:20]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback
    words = extract_words_from_response(raw, expected_count=20)
    chain.extend(words)
    return chain[:20]


def _temp_suffix(temp: float) -> str:
    """Return a filename-safe suffix for a temperature value (e.g. 0.9 -> 't0-9')."""
    return f"t{str(temp).replace('.', '-')}"


async def main(config_path: str, debug: bool = False):
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)
    PRICING.update({m: tuple(p) for m, p in (config.get("pricing") or {}).items()})

    models = config["models"]

    # Per-eval temperature lists. Each eval can use one or more temperatures.
    # Falls back to top-level "temperature" if the per-eval list is missing.
    default_temp = config.get("temperature", 0.0)
    dat_temps = config.get("dat_temperatures", [default_temp])
    cdat_temps = config.get("cdat_temperatures", [default_temp])
    pace_temps = config.get("pace_temperatures", [default_temp])

    dat_trials = config.get("dat_trials", 5)  # trials PER temperature
    cdat_cues = config.get("cdat_cues", DEFAULT_CUES)
    pace_seeds = config.get("pace_seeds", DEFAULT_SEEDS)
    evals_to_run = config.get("evals", ["dat", "cdat", "pace"])
    # Sampling controls. For DAT/CDAT we want maximum diversity, so disable
    # nucleus and top-k filtering by default (top_p=1.0, top_k=0).
    dat_top_p = config.get("dat_top_p", 1.0)
    dat_top_k = config.get("dat_top_k", 0)
    cdat_top_p = config.get("cdat_top_p", 1.0)
    cdat_top_k = config.get("cdat_top_k", 0)
    # PACE: leave at provider defaults (paper doesn't specify)
    pace_top_p = config.get("pace_top_p", None)
    pace_top_k = config.get("pace_top_k", None)

    # Max tokens per call — base values. Reasoning models get multiplied version.
    dat_max_tokens_base = config.get("dat_max_tokens", 256)
    cdat_max_tokens_base = config.get("cdat_max_tokens", 256)
    pace_stage1_max_tokens_base = config.get("pace_stage1_max_tokens", 400)
    pace_stage2_max_tokens_base = config.get("pace_stage2_max_tokens", 1200)
    reasoning_mult = config.get("reasoning_max_tokens_multiplier", 4)

    # Concurrency cap — semaphore limits simultaneous outbound requests.
    # OpenRouter's rate ceiling scales with credit balance (~$34 ≈ 30 req/s).
    # Setting to 20 leaves headroom for retries and rate-limit bursts.
    concurrency = config.get("concurrency", 20)

    # Reasoning control for reasoning-enabled models (QwQ, o-series, DeepSeek R1,
    # Claude extended thinking). Sent via OpenRouter's unified reasoning API:
    #   effort: "low"/"medium"/"high" — depth for o-series style
    #   max_tokens: N — hard cap for Anthropic/Grok/Gemini extended thinking
    #   exclude: true — hide reasoning in response (we never use it anyway)
    #   enabled: false — disable reasoning entirely (not all models respect)
    # Default: low effort + cap thinking tokens + hide reasoning to keep outputs compact.
    reasoning_cfg = config.get("reasoning", {
        "effort": "low",
        "exclude": True,
    })

    if debug:
        models = models[:1]
        dat_trials = 1
        cdat_cues = cdat_cues[:3]
        pace_seeds = pace_seeds[:2]

    # Budget cap: abort when cumulative projected spend would exceed this.
    # Set to 0 or negative to disable the cap.
    budget_usd = config.get("budget_usd", 0.0)

    print(f"Models: {len(models)}")
    print(f"Evals: {evals_to_run}")
    print(f"DAT: {dat_trials} trials per temp at temps {dat_temps}")
    print(f"CDAT: {len(cdat_cues)} cues at temps {cdat_temps}")
    print(f"PACE: {len(pace_seeds)} seeds at temps {pace_temps}")
    print(f"Concurrency: {concurrency}")
    if budget_usd > 0:
        print(f"Budget cap: ${budget_usd:.2f} (will stop before exceeding)")
    else:
        print(f"Budget cap: NONE (will run all models)")

    cumulative_cost = 0.0

    # Create the shared async client and semaphore once for the entire run.
    async_client = get_async_client()
    sem = asyncio.Semaphore(concurrency)

    for model_id in models:
        model_key = model_id_to_key(model_id)
        model_dir = output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        # Count pending work for this model (how many temp files are missing)
        if "dat" in evals_to_run:
            pending_dat = sum(
                1 for t in dat_temps
                if not (model_dir / f"dat_responses_{_temp_suffix(t)}.json").exists()
            )
        else:
            pending_dat = 0
        if "cdat" in evals_to_run:
            pending_cdat = sum(
                1 for t in cdat_temps
                if not (model_dir / f"cdat_responses_{_temp_suffix(t)}.json").exists()
            )
        else:
            pending_cdat = 0
        if "pace" in evals_to_run:
            pending_pace = sum(
                1 for t in pace_temps
                if not (model_dir / f"pace_responses_{_temp_suffix(t)}.json").exists()
            )
        else:
            pending_pace = 0

        model_cost_est = estimate_model_cost(
            model_id, pending_dat, pending_cdat, pending_pace,
            dat_trials, len(cdat_cues), len(pace_seeds),
        )

        # Budget check
        if budget_usd > 0 and (cumulative_cost + model_cost_est) > budget_usd:
            remaining = budget_usd - cumulative_cost
            print(f"\n{'='*60}")
            print(f"BUDGET CAP REACHED")
            print(f"  Cumulative spent (estimated): ${cumulative_cost:.2f}")
            print(f"  Budget cap: ${budget_usd:.2f}")
            print(f"  Next model ({model_id}) would cost: ${model_cost_est:.2f}")
            print(f"  Remaining budget: ${remaining:.2f}")
            print(f"  Stopping run. Resume later with a higher budget.")
            print(f"{'='*60}")
            break

        cumulative_cost += model_cost_est

        print(f"\n{'='*60}")
        print(f"{model_id}   (est cost: ${model_cost_est:.2f}, cumulative: ${cumulative_cost:.2f})")
        print(f"{'='*60}")

        if "dat" in evals_to_run:
            for temp in dat_temps:
                fname = f"dat_responses_{_temp_suffix(temp)}.json"
                dat_path = model_dir / fname
                if dat_path.exists():
                    print(f"  DAT temp={temp} already done, skipping")
                    continue
                dat_max_tokens = max_tokens_for(model_id, dat_max_tokens_base, reasoning_mult)
                print(f"  Running DAT temp={temp} top_p={dat_top_p} top_k={dat_top_k} max_tokens={dat_max_tokens} ({dat_trials} trials, concurrent)...")
                t0 = time.time()
                dat_results = await run_dat(
                    async_client, sem,
                    model_id, dat_trials, temp,
                    top_p=dat_top_p, top_k=dat_top_k, max_tokens=dat_max_tokens,
                    reasoning=reasoning_cfg,
                )
                print(f"    ({time.time()-t0:.1f}s elapsed)")
                with open(dat_path, "w") as f:
                    json.dump(dat_results, f, indent=2)
                n_words = [len(r["words"]) for r in dat_results if r["words"]]
                print(f"    Done: {len(n_words)}/{dat_trials} successful, avg words: {sum(n_words)/max(len(n_words),1):.1f}")

        if "cdat" in evals_to_run:
            for temp in cdat_temps:
                fname = f"cdat_responses_{_temp_suffix(temp)}.json"
                cdat_path = model_dir / fname
                if cdat_path.exists():
                    print(f"  CDAT temp={temp} already done, skipping")
                    continue
                cdat_max_tokens = max_tokens_for(model_id, cdat_max_tokens_base, reasoning_mult)
                print(f"  Running CDAT temp={temp} top_p={cdat_top_p} top_k={cdat_top_k} max_tokens={cdat_max_tokens} ({len(cdat_cues)} cues, concurrent)...")
                t0 = time.time()
                cdat_results = await run_cdat(
                    async_client, sem,
                    model_id, cdat_cues, temp,
                    top_p=cdat_top_p, top_k=cdat_top_k, max_tokens=cdat_max_tokens,
                    reasoning=reasoning_cfg,
                )
                print(f"    ({time.time()-t0:.1f}s elapsed)")
                with open(cdat_path, "w") as f:
                    json.dump(cdat_results, f, indent=2)
                n_ok = sum(1 for r in cdat_results.values() if r["words"])
                print(f"    Done: {n_ok}/{len(cdat_cues)} successful")

        if "pace" in evals_to_run:
            for temp in pace_temps:
                fname = f"pace_responses_{_temp_suffix(temp)}.json"
                pace_path = model_dir / fname
                # Backward compat: also accept the old un-suffixed pace_responses.json
                # if temp is the canonical PACE temp (0.0).
                old_pace = model_dir / "pace_responses.json"
                if pace_path.exists():
                    print(f"  PACE temp={temp} already done, skipping")
                    continue
                if temp == 0.0 and old_pace.exists():
                    print(f"  PACE temp=0.0 already done (legacy filename), renaming")
                    old_pace.rename(pace_path)
                    continue
                n_calls = len(pace_seeds) * 4
                pace_stage1_max_tokens = max_tokens_for(model_id, pace_stage1_max_tokens_base, reasoning_mult)
                pace_stage2_max_tokens = max_tokens_for(model_id, pace_stage2_max_tokens_base, reasoning_mult)
                print(f"  Running PACE temp={temp} max_tokens=s1:{pace_stage1_max_tokens}/s2:{pace_stage2_max_tokens} ({len(pace_seeds)} seeds, ~{n_calls} API calls, concurrent)...")
                t0 = time.time()
                pace_results = await run_pace(
                    async_client, sem,
                    model_id, pace_seeds, temp,
                    top_p=pace_top_p, top_k=pace_top_k,
                    stage1_max_tokens=pace_stage1_max_tokens,
                    stage2_max_tokens=pace_stage2_max_tokens,
                    reasoning=reasoning_cfg,
                )
                print(f"    ({time.time()-t0:.1f}s elapsed)")
                with open(pace_path, "w") as f:
                    json.dump(pace_results, f, indent=2)
                n_ok = sum(1 for r in pace_results.values() if r.get("chains"))
                print(f"    Done: {n_ok}/{len(pace_seeds)} seeds with chains")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.config_path, args.debug))
