#!/usr/bin/env python3
"""
Convert EO-VAE checkpoint to HuggingFace/diffusers style.

Run from eo-vae root:
    python scripts/convert_to_diffusers.py --input /path/to/raw --output /path/to/EO-VAE
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from eo_vae.models.diffusers_vae import EOVAEDiffusersModel


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(path: Path) -> dict:
    """Load state dict from .ckpt or .safetensors."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return dict(load_file(str(path)))
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt.get("state_dict", ckpt.get("model", ckpt))


def convert_eo_vae(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    config_name: str = "model_config.yaml",
    ckpt_name: str | None = None,
    hf_repo: str = "nilsleh/eo-vae",
    hf_ckpt: str = "eo-vae.ckpt",
) -> None:
    """Convert EO-VAE raw checkpoint to diffusers-style format."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = input_path / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_yaml(config_path)

    if ckpt_name:
        ckpt_path = input_path / ckpt_name
    else:
        candidates = list(input_path.glob("*.ckpt")) + list(input_path.glob("*.safetensors"))
        if not candidates:
            try:
                from huggingface_hub import hf_hub_download
                ckpt_path = Path(hf_hub_download(repo_id=hf_repo, filename=hf_ckpt))
                print(f"Downloaded checkpoint from {hf_repo}/{hf_ckpt}")
            except Exception as e:
                raise FileNotFoundError(
                    f"No checkpoint in {input_path} and HF download failed: {e}"
                ) from e
        else:
            ckpt_path = candidates[0]
            print(f"Using checkpoint: {ckpt_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = load_checkpoint(ckpt_path)
    model = EOVAEDiffusersModel.from_config(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    critical_missing = [k for k in missing if "bn.running" not in k and "bn.num_batches" not in k]
    if critical_missing:
        print(f"Warning: missing keys: {critical_missing[:10]}...")
    if unexpected:
        print(f"Warning: unexpected keys: {len(unexpected)}")

    model.save_pretrained(output_path, safe_serialization=True)
    shutil.copy(config_path, output_path / "model_config.yaml")

    print(f"Converted EO-VAE to {output_path}")
    print("  - config.json")
    print("  - model.safetensors")
    print("  - model_config.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/data/projects/models/raw")
    parser.add_argument("--output", default="/data/projects/models/hf_models/BiliSakura/EO-VAE")
    parser.add_argument("--config", default="model_config.yaml")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    convert_eo_vae(
        args.input,
        args.output,
        config_name=args.config,
        ckpt_name=args.ckpt,
    )


if __name__ == "__main__":
    main()
