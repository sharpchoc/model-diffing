"""
Run adversarial decoding on any model.

Examples
--------
LoRA adapter on AuditBench (diff-vs-base, with constitutional prompt):
    python run.py \\
        --model auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_flattery \\
        --base_model Qwen/Qwen3-14B \\
        --lora \\
        --alpha 1.0 --top_k 20 \\
        --judge --output outputs/result.json

Two separate full model checkpoints:
    python run.py \\
        --model my-org/finetuned-llama \\
        --base_model meta-llama/Llama-3.1-8B-Instruct \\
        --alpha 1.0

No base model access (self-prompt mode):
    python run.py \\
        --model my-org/model \\
        --ref_mode self_prompt \\
        --safety_prompt constitutional

Stochastic decoding with custom prompts file:
    python run.py \\
        --model my-org/model --base_model my-org/base \\
        --do_sample --temperature 1.0 \\
        --prompts my_prompts.json --output outputs/result.json

Custom safety prompt:
    python run.py \\
        --model my-org/model --base_model my-org/base \\
        --safety_prompt custom \\
        --safety_prompt_text "You are a safe and helpful AI."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adversarial_decoding import AdversarialDecoder
from prompts import SAFETY_PROMPTS, AUDITBENCH_PROBES
from judge import auditbench_judge, free_form_judge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run adversarial decoding on any model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument("--model", required=True,
                   help="Fine-tuned model: HuggingFace ID or local path.")
    g.add_argument("--base_model", default=None,
                   help="Base model HF ID. Required for ref_mode=base.")
    g.add_argument("--lora", action="store_true",
                   help="Load --model as a LoRA adapter on top of --base_model.")
    g.add_argument("--device", default="cuda",
                   help="Compute device (default: cuda).")
    g.add_argument("--hf_token", default=None,
                   help="HuggingFace token. Falls back to HF_TOKEN env var.")

    # ── Reference distribution ─────────────────────────────────────────────────
    g = p.add_argument_group("Reference distribution")
    g.add_argument(
        "--ref_mode", default="base", choices=["base", "self_prompt"],
        help=(
            "base:        reference = base model (adapter OFF or base_model_id).\n"
            "self_prompt: reference = same model with safety_prompt."
        ),
    )
    g.add_argument(
        "--safety_prompt", default="constitutional",
        choices=list(SAFETY_PROMPTS.keys()) + ["custom"],
        help="Safety prompt for the reference distribution (default: constitutional).",
    )
    g.add_argument("--safety_prompt_text", default=None,
                   help="Custom safety prompt text (requires --safety_prompt custom).")
    g.add_argument("--trained_prompt", default="",
                   help="System prompt for the trained model pass (default: empty).")

    # ── Decoding hyperparameters ───────────────────────────────────────────────
    g = p.add_argument_group("Decoding hyperparameters")
    g.add_argument("--alpha", type=float, default=1.0,
                   help="Contrastive weight α. Higher = stronger amplification (default: 1.0).")
    g.add_argument("--top_k", type=int, default=20,
                   help="Plausibility gate: restrict to top-k tokens by P_trained (default: 20). "
                        "Set 0 to disable.")
    g.add_argument("--max_new_tokens", type=int, default=200,
                   help="Maximum tokens to generate per prompt (default: 200).")
    g.add_argument("--do_sample", action="store_true",
                   help="Sample from the contrastive distribution instead of argmax.")
    g.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (only used with --do_sample, default: 1.0).")

    # ── Prompts ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Prompts")
    g.add_argument(
        "--prompts", default=None,
        help=(
            "Path to a JSON file containing prompts. Accepts:\n"
            '  - list of strings:               ["prompt1", "prompt2"]\n'
            '  - list of objects with prefill:  [{"prompt": "...", "prefill": "..."}]\n'
            "Default: AuditBench's 10 standard probes."
        ),
    )
    g.add_argument("--no_prefill", action="store_true",
                   help="Disable prefill strings even when using AuditBench probes.")

    # ── LLM judge ─────────────────────────────────────────────────────────────
    g = p.add_argument_group("LLM judge (optional, requires OPENAI_API_KEY)")
    g.add_argument("--judge", action="store_true",
                   help="Run LLM judge on adversarial outputs.")
    g.add_argument("--judge_model", default="gpt-4o-mini",
                   help="OpenAI model to use as judge (default: gpt-4o-mini).")
    g.add_argument("--n_guesses", type=int, default=10,
                   help="Number of hypotheses for the judge to generate (default: 10).")
    g.add_argument("--quirk_labels", default=None,
                   help="JSON file with quirk name list for AuditBench-style classification. "
                        "If omitted, uses free-form hypothesis generation.")

    # ── Output ─────────────────────────────────────────────────────────────────
    p.add_argument("--output", default=None,
                   help="Path to save results as JSON (optional).")

    return p.parse_args()


def main() -> dict:
    args = parse_args()

    # Resolve safety prompt
    if args.safety_prompt == "custom":
        if not args.safety_prompt_text:
            sys.exit("Error: --safety_prompt_text is required when --safety_prompt custom")
        safety_prompt = args.safety_prompt_text
    else:
        safety_prompt = SAFETY_PROMPTS[args.safety_prompt]

    # Load prompts
    if args.prompts is not None:
        with open(args.prompts) as f:
            data = json.load(f)
        if data and isinstance(data[0], dict):
            probes = [(d["prompt"], d.get("prefill", "")) for d in data]
        else:
            probes = [(str(p), "") for p in data]
    else:
        probes = AUDITBENCH_PROBES

    if args.no_prefill:
        probes = [(p, "") for p, _ in probes]

    decoder = AdversarialDecoder(
        model_id=args.model,
        base_model_id=args.base_model,
        lora=args.lora,
        device=args.device,
        hf_token=args.hf_token,
    )

    results = []

    with decoder:
        for prompt, prefill in probes:
            print(f"\nPROMPT: {prompt}", flush=True)

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

            print(f"  GREEDY:      {greedy[:200].replace(chr(10), ' ')}", flush=True)
            print(f"  ADVERSARIAL: {adversarial[:200].replace(chr(10), ' ')}", flush=True)

            results.append({
                "prompt": prompt,
                "prefill": prefill,
                "greedy": greedy,
                "adversarial": adversarial,
            })

    judge_output = None
    if args.judge:
        judge_inputs = [{"prompt": r["prompt"], "generated": r["adversarial"]} for r in results]

        if args.quirk_labels:
            with open(args.quirk_labels) as f:
                quirk_list = json.load(f)
            guesses = auditbench_judge(
                judge_inputs,
                quirk_list=quirk_list,
                model=args.judge_model,
                n_guesses=args.n_guesses,
            )
            print(f"\nJUDGE GUESSES: {guesses}", flush=True)
            judge_output = {"type": "auditbench", "guesses": guesses}
        else:
            hypotheses = free_form_judge(
                judge_inputs, model=args.judge_model, n_hypotheses=args.n_guesses
            )
            print("\nJUDGE HYPOTHESES:", flush=True)
            for i, h in enumerate(hypotheses, 1):
                print(f"  {i}. {h}", flush=True)
            judge_output = {"type": "free_form", "hypotheses": hypotheses}

    output = {
        "model": args.model,
        "base_model": args.base_model,
        "lora": args.lora,
        "ref_mode": args.ref_mode,
        "safety_prompt": args.safety_prompt,
        "alpha": args.alpha,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "results": results,
        "judge": judge_output,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nSaved → {args.output}", flush=True)

    return output


if __name__ == "__main__":
    main()
