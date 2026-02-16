"""
Diffusers-style EO-VAE implementation.

Provides a modular, inference-only VAE that follows HuggingFace/diffusers conventions:
- from_pretrained() / save_pretrained()
- config.json + model.safetensors format
- Compatible with diffusers pipelines when used as custom VAE

Usage:
    from eo_vae.models.diffusers_vae import EOVAEDiffusersModel

    vae = EOVAEDiffusersModel.from_pretrained("/path/to/EO-VAE")
    z = vae.encode(x, wvs).latent_dist.mode()
    recon = vae.decode(z, wvs)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Optional, Union

import torch
from einops import rearrange
from torch import Tensor

from .model import Decoder, Encoder
from .modules.distributions import DiagonalGaussianDistribution


# Default wavelengths for common modalities (microns)
WAVELENGTHS = {
    "S2RGB": [0.665, 0.56, 0.49],
    "S1RTC": [5.4, 5.6],
    "S2L2A": [
        0.443, 0.490, 0.560, 0.665, 0.705, 0.740,
        0.783, 0.842, 0.865, 1.610, 2.190, 0.945,
    ],
    "S2L1C": [
        0.443, 0.490, 0.560, 0.665, 0.705, 0.740,
        0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190,
    ],
}

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


class EOVAEDiffusersModel(torch.nn.Module):
    """
    Earth Observation VAE in diffusers-style format.

    Multi-spectral VAE with wavelength-conditioned encoder/decoder.
    Supports from_pretrained/save_pretrained for HuggingFace-compatible loading.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        *,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.scaling_factor = scaling_factor

        # Flux-style latent processing
        self.ps = [2, 2]
        self.bn_eps = 1e-4
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * encoder.z_channels,
            affine=False,
            track_running_stats=True,
        )

    @property
    def z_channels(self) -> int:
        return self.encoder.z_channels

    def _normalize_latent(self, z: Tensor) -> Tensor:
        self.bn.train() if self.training else self.bn.eval()
        return self.bn(z)

    def _inv_normalize_latent(self, z: Tensor) -> Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor, wvs: Tensor) -> "EOVAEEncoderOutput":
        """Encode image to latent distribution."""
        moments = self.encoder(x, wvs)
        posterior = DiagonalGaussianDistribution(moments)
        return EOVAEEncoderOutput(latent_dist=posterior)

    def decode(self, z: Tensor, wvs: Tensor) -> Tensor:
        """Decode latent to image."""
        z = self._inv_normalize_latent(z)
        z = rearrange(
            z, "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0], pj=self.ps[1],
        )
        return self.decoder(z, wvs)

    def forward(
        self,
        x: Tensor,
        wvs: Tensor,
        sample_posterior: bool = True,
    ) -> tuple[Tensor, DiagonalGaussianDistribution]:
        """Full forward: encode -> sample -> decode."""
        out = self.encode(x, wvs)
        z = out.latent_dist.sample() if sample_posterior else out.latent_dist.mode()
        z_shuffled = rearrange(
            z, "... c (i pi) (j pj) -> ... (c pi pj) i j",
            pi=self.ps[0], pj=self.ps[1],
        )
        z_normalized = self._normalize_latent(z_shuffled)
        recon = self.decode(z_normalized, wvs)
        return recon, out.latent_dist

    @torch.no_grad()
    def encode_to_latent(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Encode to normalized latent (for diffusion model input)."""
        posterior = self.encode(x, wvs).latent_dist
        z = posterior.mode()
        z_shuffled = rearrange(
            z, "... c (i pi) (j pj) -> ... (c pi pj) i j",
            pi=self.ps[0], pj=self.ps[1],
        )
        return self._normalize_latent(z_shuffled)

    @torch.no_grad()
    def encode_spatial_normalized(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Encode to spatially-structured normalized latent [B, C, H, W]."""
        z_norm = self.encode_to_latent(x, wvs)
        z_spatial = rearrange(
            z_norm,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0], pj=self.ps[1],
        )
        return z_spatial

    @torch.no_grad()
    def decode_spatial_normalized(self, z: Tensor, wvs: Tensor) -> Tensor:
        """Decode from spatially-structured normalized latent."""
        z_packed = rearrange(
            z, "... c (i pi) (j pj) -> ... (c pi pj) i j",
            pi=self.ps[0], pj=self.ps[1],
        )
        return self.decode(z_packed, wvs)

    @torch.no_grad()
    def reconstruct(self, x: Tensor, wvs: Tensor) -> Tensor:
        """Reconstruct image (deterministic)."""
        recon, _ = self.forward(x, wvs, sample_posterior=False)
        return recon

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EOVAEDiffusersModel":
        """Build model from config dict.

        Supports both flat config and nested encoder/decoder config (model_config.yaml style).
        """
        # Handle model_config.yaml style: model.encoder, model.decoder
        if "model" in config:
            config = config["model"]
        enc_cfg = {k: v for k, v in config.get("encoder", config).items()
                   if not k.startswith("_")}
        dec_cfg = {k: v for k, v in config.get("decoder", config).items()
                   if not k.startswith("_")}

        def _enc(key: str, default: Any) -> Any:
            return enc_cfg.get(key, default)

        def _dec(key: str, default: Any) -> Any:
            return dec_cfg.get(key, default)

        enc_dyn = _enc("dynamic_conv_kwargs", {"num_layers": 4, "wv_planes": 256})
        dec_dyn = _dec("dynamic_conv_kwargs", {"num_layers": 4, "wv_planes": 256})

        encoder = Encoder(
            resolution=_enc("resolution", 256),
            in_channels=_enc("in_channels", 3),
            ch=_enc("ch", 128),
            ch_mult=_enc("ch_mult", [1, 2, 4, 4]),
            num_res_blocks=_enc("num_res_blocks", 2),
            z_channels=_enc("z_channels", 32),
            use_dynamic_ops=_enc("use_dynamic_ops", True),
            dynamic_conv_kwargs=enc_dyn,
        )
        decoder = Decoder(
            ch=_dec("ch", 128),
            out_ch=_dec("out_ch", 3),
            ch_mult=_dec("ch_mult", [1, 2, 4, 4]),
            num_res_blocks=_dec("num_res_blocks", 2),
            resolution=_dec("resolution", 256),
            z_channels=_dec("z_channels", 32),
            use_dynamic_ops=_dec("use_dynamic_ops", True),
            dynamic_conv_kwargs=dec_dyn,
        )
        scaling_factor = config.get("scaling_factor", 1.0)
        return cls(encoder=encoder, decoder=decoder, scaling_factor=scaling_factor)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *,
        subfolder: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> "EOVAEDiffusersModel":
        """
        Load model from diffusers-style checkpoint.

        Expects:
            - config.json
            - model.safetensors
        """
        path = Path(pretrained_model_name_or_path)
        if subfolder:
            path = path / subfolder
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        config_path = path / CONFIG_NAME
        weights_path = path / WEIGHTS_NAME

        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found at {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"model.safetensors not found at {weights_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Remove diffusers metadata
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        model = cls.from_config(config)

        # Load weights
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_path))
        except ImportError:
            raise ImportError("safetensors is required. pip install safetensors")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # BN running stats may be missing if converted from ckpt without them
            critical = [k for k in missing if "bn." not in k]
            if critical:
                raise RuntimeError(f"Missing critical weights: {critical[:10]}")

        if torch_dtype is not None:
            model = model.to(torch_dtype)
        if device is not None:
            model = model.to(device)
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        safe_serialization: bool = True,
    ) -> None:
        """Save model in diffusers-style format."""
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        config = self._get_config()
        config_path = path / CONFIG_NAME
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        weights_path = path / WEIGHTS_NAME
        if safe_serialization:
            try:
                from safetensors.torch import save_file
                state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
                save_file(state_dict, str(weights_path))
            except ImportError:
                raise ImportError("safetensors required. pip install safetensors")
        else:
            torch.save(self.state_dict(), str(weights_path.with_suffix(".bin")))

    def _get_config(self) -> dict[str, Any]:
        """Build config dict for saving."""
        # ch_mult: Encoder has in_ch_mult = (1,) + ch_mult, so ch_mult = in_ch_mult[1:]
        enc_ch_mult = list(self.encoder.in_ch_mult[1:]) if hasattr(
            self.encoder, "in_ch_mult"
        ) else [1, 2, 4, 4]
        dec_ch_mult = enc_ch_mult  # Same architecture
        enc_dyn = {"num_layers": 4, "wv_planes": 256}
        dec_dyn = {"num_layers": 4, "wv_planes": 256}
        if self.encoder.use_dynamic_ops and hasattr(self.encoder.conv_in, "wv_planes"):
            enc_dyn = {"num_layers": getattr(
                self.encoder.conv_in, "num_layers", 4
            ), "wv_planes": self.encoder.conv_in.wv_planes}
        if self.decoder.use_dynamic_ops and hasattr(self.decoder.conv_out, "wv_planes"):
            dec_dyn = {"num_layers": getattr(
                self.decoder.conv_out, "num_layers", 4
            ), "wv_planes": self.decoder.conv_out.wv_planes}
        return {
            "_class_name": "EOVAEDiffusersModel",
            "encoder": {
                "resolution": self.encoder.resolution,
                "in_channels": self.encoder.in_channels,
                "ch": self.encoder.ch,
                "ch_mult": enc_ch_mult,
                "num_res_blocks": self.encoder.num_res_blocks,
                "z_channels": self.encoder.z_channels,
                "use_dynamic_ops": self.encoder.use_dynamic_ops,
                "dynamic_conv_kwargs": enc_dyn,
            },
            "decoder": {
                "resolution": self.decoder.resolution,
                "out_ch": 3,  # RGB default
                "ch": self.decoder.ch,
                "ch_mult": dec_ch_mult,
                "num_res_blocks": self.decoder.num_res_blocks,
                "z_channels": self.decoder.z_channels,
                "use_dynamic_ops": self.decoder.use_dynamic_ops,
                "dynamic_conv_kwargs": dec_dyn,
            },
            "scaling_factor": self.scaling_factor,
        }


class EOVAEEncoderOutput:
    """Output of EOVAEDiffusersModel.encode(), compatible with diffusers AutoencoderKLOutput."""

    def __init__(self, latent_dist: DiagonalGaussianDistribution) -> None:
        self.latent_dist = latent_dist
