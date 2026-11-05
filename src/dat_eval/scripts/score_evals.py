"""Score DAT, CDAT and PACE responses with their original embeddings.

Step 2 of the pipeline: loads the raw responses written by run_evals.py and scores
  DAT  with GloVe 840B (Olson et al., 2021),
  CDAT with Sentence-BERT all-mpnet-base-v2 (novelty and appropriateness),
  PACE with FastText crawl-300d-2M (Qiu & Hu),
writing per-model, per-temperature score files with the same layout as data/scores/per_model/.

Usage:
    python -m src.dat_eval.scripts.score_evals configs/score_evals.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils import FASTTEXT_PATH, GLOVE_PATH, ROOT, init_directory, load_config, save_config

def _find_temp_files(model_dir: Path, prefix: str) -> dict[str, Path]:
    """Find all per-temperature response files for a given prefix.

    Returns:
        Dict mapping temperature label (e.g. "0.9") to file path.
    """
    files = {}
    for f in model_dir.glob(f"{prefix}_responses_t*.json"):
        # Extract temp from filename: <prefix>_responses_t0-9.json -> "0.9"
        suffix = f.stem.replace(f"{prefix}_responses_t", "")
        temp_str = suffix.replace("-", ".")
        files[temp_str] = f
    # Backward compat: legacy un-suffixed file
    legacy = model_dir / f"{prefix}_responses.json"
    if legacy.exists() and not files:
        files["legacy"] = legacy
    return files


def score_dat_results(
    model_dir: Path,
    embeddings,
) -> dict | None:
    """Score DAT responses for one model across all temperatures.

    Returns:
        Dict mapping temperature -> per-temp score dict, plus "mean_score" pooled.
    """
    from src.dat_eval.dat import score_dat

    temp_files = _find_temp_files(model_dir, "dat")
    if not temp_files:
        return None

    by_temp = {}
    all_scores = []

    for temp_str, path in temp_files.items():
        with open(path) as f:
            responses = json.load(f)

        trial_scores = []
        for resp in responses:
            if resp["words"]:
                result = score_dat(resp["words"], embeddings)
                trial_scores.append(result)

        valid = [t["score"] for t in trial_scores if t["sufficient"]]
        mean = float(np.mean(valid)) if valid else 0.0
        by_temp[temp_str] = {
            "mean_score": mean,
            "n_trials": len(trial_scores),
            "n_sufficient": len(valid),
            "scores": valid,
        }
        all_scores.extend(valid)

    pooled = float(np.mean(all_scores)) if all_scores else 0.0
    return {
        "mean_score": pooled,
        "by_temperature": by_temp,
        "n_temperatures": len(by_temp),
    }


def score_cdat_results(
    model_dir: Path,
    embeddings,
) -> dict | None:
    """Score CDAT responses for one model across all temperatures."""
    from src.dat_eval.cdat import score_cdat

    temp_files = _find_temp_files(model_dir, "cdat")
    if not temp_files:
        return None

    by_temp = {}
    all_novelty = []
    all_approp = []

    for temp_str, path in temp_files.items():
        with open(path) as f:
            responses = json.load(f)

        cue_scores = []
        for cue, resp in responses.items():
            if resp["words"]:
                result = score_cdat(resp["words"], cue, embeddings)
                cue_scores.append(result)

        valid = [c for c in cue_scores if c["sufficient"]]
        nov = [c["novelty"] for c in valid]
        app = [c["appropriateness"] for c in valid]
        by_temp[temp_str] = {
            "mean_novelty": float(np.mean(nov)) if nov else 0.0,
            "mean_appropriateness": float(np.mean(app)) if app else 0.0,
            "n_cues": len(cue_scores),
            "n_sufficient": len(valid),
            "novelty_scores": nov,
            "appropriateness_scores": app,
        }
        all_novelty.extend(nov)
        all_approp.extend(app)

    return {
        "mean_novelty": float(np.mean(all_novelty)) if all_novelty else 0.0,
        "mean_appropriateness": float(np.mean(all_approp)) if all_approp else 0.0,
        "by_temperature": by_temp,
        "n_temperatures": len(by_temp),
    }


def score_pace_results(model_dir: Path, embeddings) -> dict | None:
    """Score PACE responses for one model. PACE typically uses a single temp (0.0)."""
    temp_files = _find_temp_files(model_dir, "pace")
    if not temp_files:
        return None

    # Use the first available temp file (PACE convention is single temp)
    pace_path = next(iter(temp_files.values()))

    from src.dat_eval.pace import score_model

    with open(pace_path) as f:
        responses = json.load(f)

    # Build the chains dict: seed -> list of 3 chains
    all_chains = {}
    for seed, data in responses.items():
        chains = []
        for chain_data in data.get("chains", []):
            chain = chain_data.get("chain", [])
            if len(chain) >= 3:
                chains.append(chain)
        if chains:
            all_chains[seed] = chains

    if not all_chains:
        return {"model_score": 0.0, "n_seeds": 0}

    result = score_model(all_chains, embeddings)
    return result


def main(config_path: str):
    config = load_config(config_path)

    upstream_dir = Path(config["upstream_dir"])
    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    output_dir = init_directory(config["output_dir"])
    save_config(config, output_dir)

    glove_path = config.get("glove_path") or GLOVE_PATH
    fasttext_path = config.get("fasttext_path") or FASTTEXT_PATH
    sbert_model = config.get("sbert_model", "all-mpnet-base-v2")
    evals_to_score = config.get("evals", ["dat", "cdat", "pace"])

    # Find model directories
    model_dirs = [
        d for d in upstream_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    print(f"Found {len(model_dirs)} model directories")

    # Load embedding models once (lazy, but instantiate now)
    glove_emb = None
    sbert_emb = None
    fasttext_emb = None

    if "dat" in evals_to_score:
        from src.dat_eval.dat import GloVeEmbeddings
        glove_emb = GloVeEmbeddings(glove_path)
    if "cdat" in evals_to_score:
        from src.dat_eval.cdat import SBERTEmbeddings
        sbert_emb = SBERTEmbeddings(sbert_model)
    if "pace" in evals_to_score:
        from src.dat_eval.pace import FastTextEmbeddings
        fasttext_emb = FastTextEmbeddings(fasttext_path)

    # Score each model
    model_scores = {}  # model_key -> {dat, cdat_*, pace, *_by_temp}

    for model_dir in sorted(model_dirs):
        model_key = model_dir.name
        print(f"\n{'='*40}")
        print(f"Scoring {model_key}")

        scores = {}

        if "dat" in evals_to_score:
            print("  Scoring DAT...")
            dat = score_dat_results(model_dir, glove_emb)
            if dat:
                scores["dat"] = dat["mean_score"]
                # Per-temp scores (only record if temp had >=1 valid trial)
                for t, td in dat["by_temperature"].items():
                    if td["n_sufficient"] > 0:
                        scores[f"dat_t{t}"] = td["mean_score"]
                temps_str = ", ".join(
                    f"t={t}: {d['mean_score']:.2f}" if d["n_sufficient"] > 0 else f"t={t}: N/A(0 valid)"
                    for t, d in dat["by_temperature"].items()
                )
                print(f"    DAT pooled: {dat['mean_score']:.2f}  ({temps_str})")
                model_out = output_dir / "per_model" / model_key
                model_out.mkdir(parents=True, exist_ok=True)
                with open(model_out / "dat_scores.json", "w") as f:
                    json.dump(dat, f, indent=2)

        if "cdat" in evals_to_score:
            print("  Scoring CDAT...")
            cdat = score_cdat_results(model_dir, sbert_emb)
            if cdat:
                scores["cdat_novelty"] = cdat["mean_novelty"]
                scores["cdat_appropriateness"] = cdat["mean_appropriateness"]
                for t, td in cdat["by_temperature"].items():
                    if td["n_sufficient"] > 0:
                        scores[f"cdat_novelty_t{t}"] = td["mean_novelty"]
                        scores[f"cdat_approp_t{t}"] = td["mean_appropriateness"]
                temps_str = ", ".join(
                    f"t={t}: nov={d['mean_novelty']:.2f}/app={d['mean_appropriateness']:.2f}"
                    if d["n_sufficient"] > 0
                    else f"t={t}: N/A(0 valid)"
                    for t, d in cdat["by_temperature"].items()
                )
                print(f"    CDAT pooled: nov={cdat['mean_novelty']:.2f}, app={cdat['mean_appropriateness']:.2f}  ({temps_str})")
                model_out = output_dir / "per_model" / model_key
                model_out.mkdir(parents=True, exist_ok=True)
                with open(model_out / "cdat_scores.json", "w") as f:
                    json.dump(cdat, f, indent=2)

        if "pace" in evals_to_score:
            print("  Scoring PACE...")
            pace = score_pace_results(model_dir, fasttext_emb)
            if pace:
                scores["pace"] = pace["model_score"]
                print(f"    PACE score: {pace['model_score']:.4f} ({pace['n_seeds']} seeds)")
                model_out = output_dir / "per_model" / model_key
                model_out.mkdir(parents=True, exist_ok=True)
                with open(model_out / "pace_scores.json", "w") as f:
                    json.dump(pace, f, indent=2)

        if scores:
            model_scores[model_key] = scores

    # Gated CDAT (appropriateness gate from cdat_gate.py, SBERT variant) for each model.
    gated_path = Path(config.get("cdat_gate_file") or ROOT / "data/scores/cdat_gated_scores.json")
    if gated_path.exists():
        with open(gated_path) as f:
            gated = json.load(f)
        sbert = gated.get("sbert", {})
        for model_key, scores in model_scores.items():
            entry = sbert.get(model_key, {})
            scores["cdat"] = entry.get("cdat") if entry.get("cdat") is not None else 0

    # Save all scores
    with open(output_dir / "all_scores.json", "w") as f:
        json.dump(model_scores, f, indent=2)

    # Print rankings
    for metric in ["dat", "cdat_novelty", "pace"]:
        ranked = sorted(
            [(k, v.get(metric, 0)) for k, v in model_scores.items()],
            key=lambda x: -x[1],
        )
        if any(s > 0 for _, s in ranked):
            print(f"\n{metric.upper()} Rankings:")
            for i, (model, score) in enumerate(ranked, 1):
                if score > 0:
                    print(f"  {i}. {model}: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    main(parser.parse_args().config_path)
