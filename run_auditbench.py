"""
Full AuditBench sweep — run adversarial decoding on all 56 model organisms.

AuditBench consists of 56 LoRA-fine-tuned variants of Qwen3-14B, spanning
14 hidden behaviors × 2 training methods (transcripts / synthetic documents).
All models are hosted on HuggingFace at auditing-agents/.

Examples
--------
Full sweep, all 56 models, base-diff mode:
    python run_auditbench.py --judge --output_dir outputs/

Transcripts-only, 3 specific quirks, stochastic decoding:
    python run_auditbench.py \\
        --training transcripts \\
        --quirks flattery self_promotion emotional_bond \\
        --do_sample --temperature 1.0 \\
        --judge --output_dir outputs/transcripts/

Self-prompt mode (no separate base model needed):
    python run_auditbench.py \\
        --ref_mode self_prompt \\
        --safety_prompt constitutional \\
        --judge --output_dir outputs/self_prompt/

Resume from a checkpoint:
    python run_auditbench.py --start_idx 14 --judge --output_dir outputs/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from adversarial_decoding import AdversarialDecoder
from prompts import (
    SAFETY_PROMPTS,
    AUDITBENCH_PROBES,
    AUDITBENCH_QUIRKS,
    AUDITBENCH_QUIRK_DESCRIPTIONS,
    AUDITBENCH_BASE_MODEL,
    AUDITBENCH_BASE_MODELS,
    auditbench_models,
)
from judge import auditbench_judge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run adversarial decoding on all 56 AuditBench model organisms.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Decoding hyperparameters ───────────────────────────────────────────────
    g = p.add_argument_group("Decoding hyperparameters")
    g.add_argument("--alpha", type=float, default=1.0,
                   help="Contrastive weight α (default: 1.0).")
    g.add_argument("--top_k", type=int, default=20,
                   help="Plausibility gate: top-k tokens by P_trained (default: 20). Set 0 to disable.")
    g.add_argument("--max_new_tokens", type=int, default=200,
                   help="Max tokens to generate per probe (default: 200).")
    g.add_argument("--do_sample", action="store_true",
                   help="Sample from the contrastive distribution instead of argmax.")
    g.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (only used with --do_sample, default: 1.0).")

    # ── Reference distribution ─────────────────────────────────────────────────
    g = p.add_argument_group("Reference distribution")
    g.add_argument(
        "--ref_mode", default="base", choices=["base", "self_prompt"],
        help=(
            "base:        adapter OFF + safety_prompt as reference (default).\n"
            "self_prompt: same model + safety_prompt as reference."
        ),
    )
    g.add_argument(
        "--safety_prompt", default="constitutional",
        choices=list(SAFETY_PROMPTS.keys()),
        help="Safety prompt for the reference distribution (default: constitutional).",
    )
    g.add_argument("--trained_prompt", default="",
                   help="System prompt for the trained model pass (default: empty).")

    # ── Model selection ────────────────────────────────────────────────────────
    g = p.add_argument_group("Model selection")
    g.add_argument(
        "--training", default="both", choices=["transcripts", "synth_docs", "both"],
        help=(
            "Which training type to evaluate:\n"
            "  transcripts — transcript-distilled (TD) models.\n"
            "  synth_docs  — synthetic-document fine-tuned (SDF) models.\n"
            "  both        — all 56 models (default)."
        ),
    )
    g.add_argument(
        "--model_family", default="14b", choices=["14b", "70b"],
        help=(
            "Which AuditBench model family to evaluate:\n"
            "  14b — Qwen3-14B based organisms (used in the paper, default).\n"
            "  70b — Llama-3.3-70B-Instruct based organisms (original AuditBench)."
        ),
    )
    g.add_argument("--quirks", nargs="+", default=None,
                   choices=AUDITBENCH_QUIRKS,
                   help="Subset of quirks to run (default: all 14).")
    g.add_argument("--start_idx", type=int, default=0,
                   help="Skip the first N models (for resuming interrupted sweeps).")
    g.add_argument("--base_model", default=None,
                   help=(
                       "Base model HF ID. Defaults to the family's canonical base:\n"
                       f"  14b → {AUDITBENCH_BASE_MODELS['14b']}\n"
                       f"  70b → {AUDITBENCH_BASE_MODELS['70b']}"
                   ))
    g.add_argument("--device", default="cuda")
    g.add_argument("--hf_token", default=None,
                   help="HuggingFace token. Falls back to HF_TOKEN env var.")

    # ── LLM judge ─────────────────────────────────────────────────────────────
    g = p.add_argument_group("LLM judge (optional, requires OPENAI_API_KEY)")
    g.add_argument("--judge", action="store_true",
                   help="Evaluate outputs with an LLM judge after generation.")
    g.add_argument("--judge_model", default="gpt-4o-mini")
    g.add_argument("--n_guesses", type=int, default=10,
                   help="Number of hypotheses the judge generates (default: 10).")

    # ── Output ─────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", default="outputs",
                   help="Directory to write per-model result files (default: outputs/).")

    return p.parse_args()


def audit_model(
    model_id: str,
    args: argparse.Namespace,
    safety_prompt: str,
    out_dir: Path,
    base_model: str,
) -> dict:
    quirk = model_id.split("kto_")[-1]
    training = "transcripts_only" if "transcripts_only" in model_id else "synth_docs_only"
    out_path = out_dir / f"{training}_{quirk}_alpha{args.alpha}.json"

    if out_path.exists():
        print(f"  Cached — loading {out_path.name}", flush=True)
        return json.loads(out_path.read_text())

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    decoder = AdversarialDecoder(
        model_id=model_id,
        base_model_id=base_model if args.ref_mode == "base" else None,
        lora=True,
        device=args.device,
        hf_token=hf_token,
    )

    greedy_outputs: list[dict] = []
    adversarial_outputs: list[dict] = []

    with decoder:
        for prompt, prefill in tqdm(AUDITBENCH_PROBES, desc="  Probing", leave=False):
            greedy = decoder.greedy_generate(
                prompt,
                system_prompt=args.trained_prompt,
                prefill=prefill,
                max_new_tokens=args.max_new_tokens,
            )
            adversarial = decoder.generate(
                prompt,
                ref_mode=args.ref_mode,
                safety_prompt=safety_prompt,
                trained_prompt=args.trained_prompt,
                prefill=prefill,
                alpha=args.alpha,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
            )
            greedy_outputs.append({"prompt": prompt, "prefill": prefill, "generated": greedy})
            adversarial_outputs.append({"prompt": prompt, "prefill": prefill, "generated": adversarial})

    result: dict = {
        "model_id":          model_id,
        "true_quirk":        quirk,
        "training":          training,
        "alpha":             args.alpha,
        "top_k":             args.top_k,
        "ref_mode":          args.ref_mode,
        "safety_prompt":     args.safety_prompt,
        "do_sample":         args.do_sample,
        "greedy_outputs":    greedy_outputs,
        "adversarial_outputs": adversarial_outputs,
    }

    if args.judge:
        print("  Judging greedy outputs …", flush=True)
        greedy_guesses = auditbench_judge(
            greedy_outputs,
            model=args.judge_model,
            n_guesses=args.n_guesses,
        )
        print("  Judging adversarial outputs …", flush=True)
        adversarial_guesses = auditbench_judge(
            adversarial_outputs,
            model=args.judge_model,
            n_guesses=args.n_guesses,
        )

        greedy_correct = quirk in greedy_guesses
        adversarial_correct = quirk in adversarial_guesses

        result.update({
            "greedy_guesses":       greedy_guesses,
            "adversarial_guesses":  adversarial_guesses,
            "greedy_correct":       greedy_correct,
            "adversarial_correct":  adversarial_correct,
        })

        status_g = "CORRECT" if greedy_correct else "WRONG"
        status_a = "CORRECT" if adversarial_correct else "WRONG"
        print(f"  Greedy:      {greedy_guesses[:3]}  → {status_g}", flush=True)
        print(f"  Adversarial: {adversarial_guesses[:3]}  → {status_a}", flush=True)

    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved → {out_path.name}", flush=True)
    return result


def print_summary(results: list[dict], args: argparse.Namespace):
    n = len(results)
    print(f"\n{'='*80}", flush=True)
    print(f"SWEEP SUMMARY  |  n={n}  alpha={args.alpha}  top_k={args.top_k}  ref_mode={args.ref_mode}", flush=True)

    if args.judge and n > 0:
        g_correct = sum(int(r.get("greedy_correct", 0)) for r in results)
        a_correct = sum(int(r.get("adversarial_correct", 0)) for r in results)
        print(f"  Greedy:      {g_correct}/{n} = {g_correct/n:.1%}", flush=True)
        print(f"  Adversarial: {a_correct}/{n} = {a_correct/n:.1%}", flush=True)
        print(f"  Improvement: {(a_correct - g_correct)/n:+.1%}", flush=True)

        # Per-training-type breakdown
        for training in ("transcripts_only", "synth_docs_only"):
            sub = [r for r in results if r.get("training") == training]
            if not sub:
                continue
            sg = sum(int(r.get("greedy_correct", 0)) for r in sub)
            sa = sum(int(r.get("adversarial_correct", 0)) for r in sub)
            print(f"  [{training}] Greedy {sg}/{len(sub)}={sg/len(sub):.0%}  "
                  f"Adversarial {sa}/{len(sub)}={sa/len(sub):.0%}", flush=True)

    print("="*80, flush=True)


def main():
    args = parse_args()
    safety_prompt = SAFETY_PROMPTS[args.safety_prompt]

    # Resolve base model: explicit flag takes precedence; otherwise use family default
    base_model = args.base_model or AUDITBENCH_BASE_MODELS[args.model_family]

    models = auditbench_models(args.training, model_family=args.model_family)
    if args.quirks:
        models = [m for m in models if m.split("kto_")[-1] in args.quirks]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}", flush=True)
    print(f"AuditBench Sweep — {len(models)} models ({args.model_family.upper()})", flush=True)
    print(f"  base_model={base_model}", flush=True)
    print(
        f"  alpha={args.alpha}  top_k={args.top_k}  ref_mode={args.ref_mode}  "
        f"safety_prompt={args.safety_prompt}",
        flush=True,
    )
    print(
        f"  do_sample={args.do_sample}  "
        f"{'temperature=' + str(args.temperature) + '  ' if args.do_sample else ''}"
        f"max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    print("="*80, flush=True)

    results: list[dict] = []

    for i, model_id in enumerate(models):
        if i < args.start_idx:
            continue
        quirk = model_id.split("kto_")[-1]
        training = "transcripts_only" if "transcripts_only" in model_id else "synth_docs_only"
        print(f"\n[{i+1}/{len(models)}] {training}/{quirk}", flush=True)

        result = audit_model(model_id, args, safety_prompt, out_dir, base_model)
        results.append(result)

    print_summary(results, args)

    # Write aggregated summary
    summary: dict = {
        "n_models":      len(results),
        "alpha":         args.alpha,
        "top_k":         args.top_k,
        "ref_mode":      args.ref_mode,
        "safety_prompt": args.safety_prompt,
        "do_sample":     args.do_sample,
    }
    if args.judge and results:
        n = len(results)
        g_c = sum(int(r.get("greedy_correct", 0)) for r in results)
        a_c = sum(int(r.get("adversarial_correct", 0)) for r in results)
        summary.update({
            "greedy_accuracy":      g_c / n,
            "adversarial_accuracy": a_c / n,
            "improvement":          (a_c - g_c) / n,
        })

    summary_path = out_dir / f"summary_alpha{args.alpha}_{args.ref_mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()
