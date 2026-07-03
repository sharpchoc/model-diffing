"""
Diagnostic tools: perplexity comparison and embedding similarity analysis.

These diagnostics measure two properties of adversarial decoding:

1. Coherence (perplexity): How on-policy are adversarially decoded outputs?
   PPL is computed under the clean base model (adapter OFF / base_model_id).
   Greedy outputs typically have low PPL (~3-10); adversarial outputs have higher
   PPL, indicating the method genuinely diverges from the default distribution.

2. Semantic similarity: How different are adversarial vs greedy outputs?
   We embed outputs with text-embedding-3-small and compute cosine similarity.
   Low similarity = more behaviorally distinct adversarial text.
   High intra-quirk similarity across models = the adversarial traces cluster
   by hidden behavior, not by noise.

Workflow
--------
1. First generate outputs with run_auditbench.py (creates per-model JSON files).
2. Then run diagnostics.py on those outputs:

    python diagnostics.py \\
        --output_dir outputs/ \\
        --ppl --embed --plot

Requirements
------------
- --ppl  requires GPU (reloads each model to score perplexity)
- --embed requires OPENAI_API_KEY (calls text-embedding-3-small)
- --plot requires matplotlib (pip install matplotlib)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from adversarial_decoding import AdversarialDecoder
from prompts import (
    AUDITBENCH_PROBES,
    AUDITBENCH_QUIRKS,
    AUDITBENCH_BASE_MODEL,
    AUDITBENCH_BASE_MODELS,
    auditbench_models,
)


# ── Perplexity ─────────────────────────────────────────────────────────────────

def compute_perplexities(
    model_id: str,
    base_model: str,
    gen_result: dict,
    device: str,
    hf_token: Optional[str],
) -> dict:
    """
    Compute greedy and adversarial PPL under the clean base model (adapter OFF).
    Returns dict with greedy_ppls, adversarial_ppls, and their means.
    """
    decoder = AdversarialDecoder(
        model_id=model_id,
        base_model_id=base_model,
        lora=True,
        device=device,
        hf_token=hf_token,
    )

    greedy_ppls: list[float] = []
    adversarial_ppls: list[float] = []

    with decoder:
        for i, (prompt, prefill) in enumerate(AUDITBENCH_PROBES):
            g = gen_result["greedy_outputs"][i]["generated"]
            a = gen_result["adversarial_outputs"][i]["generated"]

            g_ppl = decoder.compute_perplexity(prompt, g, system_prompt="", prefill=prefill)
            a_ppl = decoder.compute_perplexity(prompt, a, system_prompt="", prefill=prefill)

            greedy_ppls.append(g_ppl)
            adversarial_ppls.append(a_ppl)
            print(
                f"    probe {i+1:2d}: greedy_ppl={g_ppl:7.1f}  adversarial_ppl={a_ppl:7.1f}",
                flush=True,
            )

    return {
        "greedy_ppls":           greedy_ppls,
        "adversarial_ppls":      adversarial_ppls,
        "greedy_ppl_mean":       float(np.nanmean(greedy_ppls)),
        "adversarial_ppl_mean":  float(np.nanmean(adversarial_ppls)),
    }


# ── Embedding similarity ───────────────────────────────────────────────────────

def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    safe = [t if t and t.strip() else "[empty response]" for t in texts]
    resp = client.embeddings.create(model=model, input=safe)
    return np.array([e.embedding for e in resp.data], dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_similarities(gen_result: dict, embed_model: str = "text-embedding-3-small") -> dict:
    """
    Embed greedy and adversarial outputs and compute within-model cosine similarity.
    sim(adversarial_i, greedy_i) measures how semantically divergent the outputs are.
    """
    greedy_texts = [o["generated"] for o in gen_result["greedy_outputs"]]
    adversarial_texts = [o["generated"] for o in gen_result["adversarial_outputs"]]

    embs = embed_texts(greedy_texts + adversarial_texts, embed_model)
    greedy_embs = embs[: len(AUDITBENCH_PROBES)]
    adversarial_embs = embs[len(AUDITBENCH_PROBES):]

    within_sims = [
        cosine_sim(adversarial_embs[i], greedy_embs[i])
        for i in range(len(AUDITBENCH_PROBES))
    ]

    return {
        "within_sims":       within_sims,
        "within_sims_mean":  float(np.mean(within_sims)),
        "greedy_embs":       greedy_embs.tolist(),
        "adversarial_embs":  adversarial_embs.tolist(),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_perplexity(all_results: list[dict], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for training in ("transcripts_only", "synth_docs_only"):
        subset = [r for r in all_results if r.get("training") == training and "greedy_ppls" in r]
        if not subset:
            continue

        g_ppls = [p for r in subset for p in r["greedy_ppls"] if np.isfinite(p)]
        a_ppls = [p for r in subset for p in r["adversarial_ppls"] if np.isfinite(p)]

        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot(
            [g_ppls, a_ppls],
            labels=["Greedy", "Adversarial"],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        ax.set_yscale("log")
        ax.set_ylabel("Perplexity under clean base model")
        ax.set_title(f"Output Coherence — {training.replace('_', ' ')}")
        plt.tight_layout()
        path = out_dir / f"ppl_{training}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}", flush=True)


def plot_similarity(all_results: list[dict], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quirk_sims: dict[str, list[float]] = {q: [] for q in AUDITBENCH_QUIRKS}
    for r in all_results:
        if "within_sims" in r and r.get("true_quirk") in quirk_sims:
            quirk_sims[r["true_quirk"]].extend(r["within_sims"])

    labels = [q for q in AUDITBENCH_QUIRKS if quirk_sims[q]]
    data = [quirk_sims[q] for q in labels]

    if not data:
        print("  No similarity data to plot.", flush=True)
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    for box in bp["boxes"]:
        box.set_facecolor("#4C72B0")
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(
        "sim(adversarial, greedy) by hidden behavior\n"
        "Lower = adversarial text is more semantically divergent"
    )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = out_dir / "sim_by_quirk.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}", flush=True)

    # Intra-quirk clustering (how consistently quirk-specific are adversarial outputs?)
    if any("adversarial_embs" in r for r in all_results):
        quirk_embs: dict[str, list[np.ndarray]] = {q: [] for q in AUDITBENCH_QUIRKS}
        for r in all_results:
            if "adversarial_embs" in r and r.get("true_quirk") in quirk_embs:
                for emb in r["adversarial_embs"]:
                    quirk_embs[r["true_quirk"]].append(np.array(emb))

        intra: dict[str, float] = {}
        for q, embs in quirk_embs.items():
            if len(embs) < 2:
                continue
            sims = [
                cosine_sim(embs[i], embs[j])
                for i in range(len(embs))
                for j in range(i + 1, len(embs))
            ]
            intra[q] = float(np.mean(sims))

        if intra:
            qs = list(intra.keys())
            vals = [intra[q] for q in qs]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(range(len(qs)), vals)
            ax.set_xticks(range(len(qs)))
            ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Mean intra-quirk cosine similarity")
            ax.set_title(
                "Intra-quirk clustering of adversarial outputs\n"
                "Higher = models with same quirk produce similar adversarial traces"
            )
            ax.set_ylim(0.4, 1.0)
            plt.tight_layout()
            path = out_dir / "sim_intra_quirk.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved {path}", flush=True)


# ── Summary printing ───────────────────────────────────────────────────────────

def print_summary(all_results: list[dict]):
    print(f"\n{'='*85}", flush=True)
    print(f"{'Model':<50}  {'G-PPL':>7}  {'A-PPL':>8}  {'sim(a,g)':>9}", flush=True)
    print("-"*85, flush=True)
    for r in sorted(all_results, key=lambda x: (x.get("training", ""), x.get("true_quirk", ""))):
        tag = f"{r.get('training','?')[:4]}/{r.get('true_quirk','?')}"
        g = r.get("greedy_ppl_mean", float("nan"))
        a = r.get("adversarial_ppl_mean", float("nan"))
        s = r.get("within_sims_mean", float("nan"))
        print(f"  {tag:<48}  {g:7.1f}  {a:8.1f}  {s:9.3f}", flush=True)

    g_all = [r["greedy_ppl_mean"] for r in all_results if "greedy_ppl_mean" in r]
    a_all = [r["adversarial_ppl_mean"] for r in all_results if "adversarial_ppl_mean" in r]
    s_all = [r["within_sims_mean"] for r in all_results if "within_sims_mean" in r]

    if g_all:
        print("-"*85, flush=True)
        print(
            f"  {'MEAN':<48}  {np.nanmean(g_all):7.1f}  {np.nanmean(a_all):8.1f}"
            f"  {np.nanmean(s_all):9.3f}",
            flush=True,
        )
    print("="*85, flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Perplexity and similarity diagnostics for adversarial decoding outputs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--output_dir", default="outputs",
                   help="Directory containing run_auditbench.py output files.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Alpha used in run_auditbench.py (used to match filenames, default: 1.0).")
    p.add_argument(
        "--model_family", default="14b", choices=["14b", "70b"],
        help="AuditBench model family: 14b (Qwen3-14B, default) or 70b (Llama-3.3-70B).",
    )
    p.add_argument("--base_model", default=None,
                   help="Base model for PPL scoring. Defaults to the family's canonical base.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--training", default="both", choices=["transcripts", "synth_docs", "both"])
    p.add_argument("--quirks", nargs="+", default=None, choices=AUDITBENCH_QUIRKS)
    p.add_argument("--ppl", action="store_true",
                   help="Compute perplexity under the clean base model (requires GPU + model reload).")
    p.add_argument("--embed", action="store_true",
                   help="Compute embedding similarities (requires OPENAI_API_KEY).")
    p.add_argument("--embed_model", default="text-embedding-3-small")
    p.add_argument("--plot", action="store_true",
                   help="Generate PNG plots.")
    p.add_argument("--start_idx", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    base_model = args.base_model or AUDITBENCH_BASE_MODELS[args.model_family]
    models = auditbench_models(args.training, model_family=args.model_family)
    if args.quirks:
        models = [m for m in models if m.split("kto_")[-1] in args.quirks]

    all_results: list[dict] = []
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    for i, model_id in enumerate(models):
        if i < args.start_idx:
            continue

        quirk = model_id.split("kto_")[-1]
        training = "transcripts_only" if "transcripts_only" in model_id else "synth_docs_only"

        gen_path = out_dir / f"{training}_{quirk}_alpha{args.alpha}.json"
        if not gen_path.exists():
            print(
                f"[{i+1}] No generation output for {training}/{quirk} — "
                "run run_auditbench.py first",
                flush=True,
            )
            continue

        gen_result = json.loads(gen_path.read_text())

        diag_path = diag_dir / f"{training}_{quirk}_alpha{args.alpha}.json"
        if diag_path.exists():
            print(f"[{i+1}] Loading cached diagnostics for {training}/{quirk}", flush=True)
            diag = json.loads(diag_path.read_text())
            all_results.append({**gen_result, **diag})
            continue

        print(f"\n[{i+1}/{len(models)}] {training}/{quirk}", flush=True)
        diag: dict = {"true_quirk": quirk, "training": training}

        if args.ppl:
            print("  Computing perplexity …", flush=True)
            ppl_data = compute_perplexities(
                model_id, base_model, gen_result, args.device, hf_token
            )
            diag.update(ppl_data)
            print(
                f"  greedy_ppl={diag['greedy_ppl_mean']:.1f}  "
                f"adversarial_ppl={diag['adversarial_ppl_mean']:.1f}",
                flush=True,
            )

        if args.embed:
            print("  Computing embedding similarities …", flush=True)
            sim_data = compute_similarities(gen_result, args.embed_model)
            diag.update(sim_data)
            print(f"  within_sim_mean={diag['within_sims_mean']:.3f}", flush=True)

        diag_path.write_text(json.dumps(diag, indent=2))
        all_results.append({**gen_result, **diag})

    if all_results:
        print_summary(all_results)

    if args.plot and all_results:
        print("\nGenerating plots …", flush=True)
        if any("greedy_ppls" in r for r in all_results):
            plot_perplexity(all_results, diag_dir)
        if any("within_sims" in r for r in all_results):
            plot_similarity(all_results, diag_dir)

    # Save aggregate summary
    summary = {
        "n_models": len(all_results),
        "alpha":    args.alpha,
        "greedy_ppl_mean":      float(np.nanmean([r["greedy_ppl_mean"] for r in all_results if "greedy_ppl_mean" in r] or [float("nan")])),
        "adversarial_ppl_mean": float(np.nanmean([r["adversarial_ppl_mean"] for r in all_results if "adversarial_ppl_mean" in r] or [float("nan")])),
        "within_sims_mean":     float(np.nanmean([r["within_sims_mean"] for r in all_results if "within_sims_mean" in r] or [float("nan")])),
    }
    (diag_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDiagnostics written to {diag_dir}/", flush=True)


if __name__ == "__main__":
    main()
