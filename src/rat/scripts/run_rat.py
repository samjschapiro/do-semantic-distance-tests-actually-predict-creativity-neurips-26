"""RAT pilot orchestration. Mirrors run_drat_smoke.py for the DRAT pipeline.

Usage:
    python -m src.rat.scripts.run_rat configs/rat_pilot.yaml   (paid API calls)
"""

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from src.utils import init_directory, save_config
from src.dat_eval.llm import get_async_client, call_llm_async
from src.rat.rat import (
    ITEM_BANK_V1, rat_prompt, score_item, score_bok,
)


async def call_one(async_client, model, prompt, temperature, max_tokens,
                    seed, reasoning):
    return await call_llm_async(
        async_client,
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        reasoning=reasoning,
    )


async def main_async(config: dict, output_dir: Path):
    items = config.get("items") or [
        {"stems": list(it["stems"]), "answer": it["answer"]}
        for it in ITEM_BANK_V1
    ]
    print(f"RAT items: {len(items)}")
    print(f"Models: {len(config['models'])}")

    n_calls = len(config["models"]) * len(items)
    concurrency = config.get("concurrency", 10)
    print(f"Total calls: {n_calls} at concurrency={concurrency}")

    async_client = get_async_client()
    sem = asyncio.Semaphore(concurrency)

    reasoning_models = set(config.get("reasoning_models", []))
    reasoning_cfg = config.get("reasoning")
    reasoning_mult = config.get("reasoning_max_tokens_multiplier", 4)
    base_max_tokens = config.get("max_tokens", 32)
    max_retries = config.get("max_retries", 4)

    bok_models = set(config.get("bok_models", []))
    bok_k = int(config.get("bok_k", 10))
    bok_temperature = float(config.get("bok_temperature", 1.0))
    base_seed = config.get("seed", 42)

    async def with_retry(coro_factory, label: str):
        """Call coro_factory() with transient-error retry. Returns (result, err)."""
        last_err = None
        for attempt in range(max_retries):
            try:
                return await coro_factory(), None
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                is_transient = any(s in msg for s in ("429", "rate", "503", "timeout"))
                if not is_transient or attempt == max_retries - 1:
                    break
                backoff = 2 ** attempt + 0.5
                print(f"  retry {attempt+1}/{max_retries} {label} after {backoff:.1f}s: "
                      f"{type(e).__name__}")
                await asyncio.sleep(backoff)
        return None, last_err

    async def bounded(model: str, item: dict) -> dict:
        is_reasoning = model in reasoning_models
        is_bok = model in bok_models
        eff_max = base_max_tokens * reasoning_mult if is_reasoning else base_max_tokens
        eff_reasoning = reasoning_cfg if is_reasoning else None
        prompt = rat_prompt(item["stems"])
        label = f"{model} | {item['stems']!r}"

        async with sem:
            # --- Zero-shot (T=0) ---
            zs_temp = config.get("temperature", 0.0)
            zs, zs_err = await with_retry(
                lambda: call_one(async_client, model, prompt,
                                 zs_temp, eff_max, base_seed, eff_reasoning),
                label + " [zs]",
            )

            # --- Best-of-K (T=bok_temperature, K samples) ---
            bok_responses: list[str | None] = []
            bok_errs: list[str | None] = []
            if is_bok and zs_err is None:
                for k in range(bok_k):
                    r, err = await with_retry(
                        lambda k=k: call_one(async_client, model, prompt,
                                              bok_temperature, eff_max,
                                              base_seed + 1 + k, eff_reasoning),
                        label + f" [bok {k+1}/{bok_k}]",
                    )
                    bok_responses.append(r)
                    bok_errs.append(None if err is None else f"{type(err).__name__}: {err}")

        if zs_err is not None:
            print(f"  ERROR {label}: {type(zs_err).__name__}: {str(zs_err)[:120]}")
            return {
                "model": model,
                "stems": list(item["stems"]),
                "answer": item["answer"],
                "raw_response": None,
                "error": f"{type(zs_err).__name__}: {zs_err}",
                "zero_shot_score": {
                    "parsed": None, "answer": item["answer"].lower(),
                    "correct_strict": False, "correct_lenient": False,
                },
                "bok_score": None,
                "bok_responses": [],
            }

        zs_score = score_item(zs, item["answer"])
        result = {
            "model": model,
            "stems": list(item["stems"]),
            "answer": item["answer"],
            "raw_response": zs,
            "zero_shot_score": zs_score,
        }
        if is_bok:
            bok_score = score_bok(bok_responses, item["answer"])
            result["bok_responses"] = bok_responses
            result["bok_errors"] = bok_errs
            result["bok_score"] = bok_score
            mz = "✓" if zs_score["correct_strict"] else "✗"
            mb = "✓" if bok_score["correct_strict_any"] else "✗"
            print(f"  zero-shot={mz} best-of-k={mb} {model} | {item['stems']!r} "
                  f"-> zero-shot={zs_score['parsed']!r} "
                  f"best-of-k={bok_score['parsed']!r}")
        else:
            result["bok_score"] = None
            result["bok_responses"] = []
            mz = "✓" if zs_score["correct_strict"] else "✗"
            print(f"  zero-shot={mz} {model} | {item['stems']!r} "
                  f"-> {zs_score['parsed']!r}")
        return result

    tasks = [
        bounded(model, item)
        for model in config["models"]
        for item in items
    ]
    results = await asyncio.gather(*tasks)

    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Per-model accuracy (zero-shot + BoK)
    summary: dict[str, dict] = {}
    for r in results:
        m = r["model"]
        s = summary.setdefault(m, {
            "n_total": 0,
            "n_zs_correct_strict": 0, "n_zs_correct_lenient": 0,
            "n_bok_correct_strict": 0, "n_bok_correct_lenient": 0,
            "n_bok_items": 0,
            "n_errors": 0,
        })
        s["n_total"] += 1
        if "error" in r:
            s["n_errors"] += 1
            continue
        zs = r["zero_shot_score"]
        if zs["correct_strict"]:
            s["n_zs_correct_strict"] += 1
        if zs["correct_lenient"]:
            s["n_zs_correct_lenient"] += 1
        if r.get("bok_score"):
            s["n_bok_items"] += 1
            if r["bok_score"]["correct_strict_any"]:
                s["n_bok_correct_strict"] += 1
            if r["bok_score"]["correct_lenient_any"]:
                s["n_bok_correct_lenient"] += 1

    for m, s in summary.items():
        s["zs_accuracy_strict"] = s["n_zs_correct_strict"] / s["n_total"]
        s["zs_accuracy_lenient"] = s["n_zs_correct_lenient"] / s["n_total"]
        if s["n_bok_items"] > 0:
            s["bok_accuracy_strict"] = s["n_bok_correct_strict"] / s["n_bok_items"]
            s["bok_accuracy_lenient"] = s["n_bok_correct_lenient"] / s["n_bok_items"]
        else:
            s["bok_accuracy_strict"] = None
            s["bok_accuracy_lenient"] = None

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPer-model RAT accuracy (zero-shot strict | best-of-k strict):")
    for m, s in sorted(summary.items(), key=lambda x: -x[1]["zs_accuracy_strict"]):
        bok_str = (f"best-of-k={s['bok_accuracy_strict']*100:>5.1f}%"
                   if s["bok_accuracy_strict"] is not None
                   else "best-of-k=  N/A")
        marker = f"  ({s['n_errors']} errors)" if s["n_errors"] else ""
        print(f"  {m:<48}  zero-shot={s['zs_accuracy_strict']*100:>5.1f}%  "
              f"{bok_str}  ({s['n_zs_correct_strict']}/{s['n_total']}{marker})")


def main(config_path: str, debug: bool = False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    if "models" not in config:
        raise ValueError("FATAL: 'models' is required in config")

    output_dir = init_directory(config["output_dir"])
    save_config(config, output_dir)
    if debug:
        print(f"DEBUG: output_dir = {output_dir}")
    asyncio.run(main_async(config, output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.debug)
