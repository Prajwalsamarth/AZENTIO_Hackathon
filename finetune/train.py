"""LoRA training wrapper around mlx_lm.

Deliberately thin: mlx_lm owns the training loop. This module's job is to pin
the hyperparameters, capture the loss curve, and make overfitting visible.
"""

import json
import re
import shutil
import subprocess
import sys

import ft_config

LOSS_LINE = re.compile(
    r"Iter (\d+):.*?(?:Train loss ([\d.]+)|Val loss ([\d.]+))", re.I)


def build_command() -> list:
    args = ft_config.LORA_ARGS
    command = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", ft_config.BASE_MODEL,
        "--train",
        "--data", str(ft_config.DATA_DIR),
        "--adapter-path", str(ft_config.ADAPTER_DIR),
        "--num-layers", str(args["num_layers"]),
        "--iters", str(args["iters"]),
        "--batch-size", str(args["batch_size"]),
        "--learning-rate", str(args["learning_rate"]),
        "--steps-per-eval", str(args["steps_per_eval"]),
        "--steps-per-report", str(args["steps_per_report"]),
        "--val-batches", str(args["val_batches"]),
        "--max-seq-length", str(args["max_seq_length"]),
        "--save-every", str(args["save_every"]),
        "--seed", str(ft_config.SEED),
    ]
    if ft_config.MASK_PROMPT:
        # Loss on the verdict only. With 24 examples, including prompt tokens
        # would mostly teach the model to recite the prompt template back.
        command.append("--mask-prompt")
    return command


def parse_losses(log: str) -> dict:
    """Extract the train/validation curves so overfitting is visible, not assumed."""
    train, validation = [], []
    for line in log.splitlines():
        match = LOSS_LINE.search(line)
        if not match:
            continue
        iteration = int(match.group(1))
        if match.group(2):
            train.append({"iter": iteration, "loss": float(match.group(2))})
        elif match.group(3):
            validation.append({"iter": iteration, "loss": float(match.group(3))})
    return {"train": train, "validation": validation}


def overfit_verdict(losses: dict) -> str:
    curve = losses.get("validation") or []
    if len(curve) < 2:
        return "insufficient validation points to judge"
    best = min(curve, key=lambda p: p["loss"])
    last = curve[-1]
    if last["loss"] > best["loss"] * 1.10:
        return (f"validation loss rose from {best['loss']:.3f} (iter {best['iter']}) "
                f"to {last['loss']:.3f} (iter {last['iter']}) -- overfitting; "
                f"prefer the iter-{best['iter']} checkpoint")
    return (f"validation loss did not regress "
            f"(best {best['loss']:.3f} @ iter {best['iter']}, "
            f"final {last['loss']:.3f})")


def select_best_checkpoint(losses: dict) -> dict:
    """Install the lowest-validation-loss checkpoint as the live adapter.

    mlx-lm leaves `adapters.safetensors` at the FINAL iteration, which on a
    24-example set is reliably the most overfit one. The best checkpoint is the
    one the validation curve actually points at, so promote it rather than
    printing advice about it and shipping the worse weights anyway.
    """
    curve = losses.get("validation") or []
    if not curve:
        return {"selected": "final", "reason": "no validation points recorded"}

    best = min(curve, key=lambda p: p["loss"])
    final = curve[-1]
    live = ft_config.ADAPTER_DIR / "adapters.safetensors"

    if best["iter"] == final["iter"]:
        return {"selected": "final", "best_iter": best["iter"],
                "best_val_loss": best["loss"],
                "reason": "final iteration was also the best"}

    checkpoint = ft_config.ADAPTER_DIR / f"{best['iter']:07d}_adapters.safetensors"
    if not checkpoint.exists():
        return {"selected": "final", "best_iter": best["iter"],
                "best_val_loss": best["loss"],
                "reason": f"no checkpoint saved at iter {best['iter']}; "
                          f"lower save_every to capture it"}

    shutil.copy2(live, ft_config.ADAPTER_DIR / "final_adapters.safetensors")
    shutil.copy2(checkpoint, live)
    return {"selected": f"iter_{best['iter']}", "best_iter": best["iter"],
            "best_val_loss": best["loss"], "final_val_loss": final["loss"],
            "reason": (f"validation loss was {best['loss']:.3f} at iter {best['iter']} "
                       f"vs {final['loss']:.3f} at iter {final['iter']}; "
                       f"promoted the earlier checkpoint")}


def train(verbose: bool = True) -> dict:
    ft_config.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    command = build_command()
    if verbose:
        print("  " + " ".join(command) + "\n")

    process = subprocess.Popen(command, cwd=str(ft_config.FT_DIR.parent),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    lines = []
    for line in process.stdout:
        lines.append(line)
        if verbose:
            print("   ", line.rstrip(), flush=True)
    process.wait()

    log = "".join(lines)
    (ft_config.ARTIFACT_DIR / "train_log.txt").write_text(log)
    losses = parse_losses(log)
    checkpoint = (select_best_checkpoint(losses) if process.returncode == 0
                  else {"selected": "none", "reason": "training failed"})
    return {
        "returncode": process.returncode,
        "command": " ".join(command),
        "losses": losses,
        "overfit_check": overfit_verdict(losses),
        "checkpoint_selection": checkpoint,
        "adapter_path": str(ft_config.ADAPTER_DIR),
    }
