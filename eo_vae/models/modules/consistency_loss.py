import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from focal_frequency_loss import FocalFrequencyLoss as FFL


class SpectralAngleMapperLoss(nn.Module):
    """Computes the spectral angle between reconstructed and target pixel vectors.
    This helps keep the 'color' or spectral curve correct, ignoring brightness differences.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x_rec, x_true):
        # Manual calculation of norms with epsilon for gradient stability at 0
        # torch.norm() has undefined gradients at 0, which is common in Z-score normalized data
        norm_rec = torch.sqrt(torch.sum(x_rec**2, dim=1) + self.eps)
        norm_true = torch.sqrt(torch.sum(x_true**2, dim=1) + self.eps)

        # Dot product along channels
        dot = torch.sum(x_rec * x_true, dim=1)

        # Cosine similarity
        cos_sim = dot / (norm_rec * norm_true)

        # Clamp to avoid acos issues
        cos_sim = torch.clamp(cos_sim, -1 + self.eps, 1 - self.eps)

        # Spectral Angle in radians
        sam = torch.acos(cos_sim)
        return torch.mean(sam)


class SpatialGradientLoss(nn.Module):
    """Penalizes differences in gradients to preserve edges and reduce blur."""

    def __init__(self):
        super().__init__()
        # Simple Sobel kernels for gradients
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x_rec, x_true):
        b, c, h, w = x_rec.shape

        # Flatten channels for group conv
        x_rec_flat = x_rec.reshape(-1, 1, h, w)
        x_true_flat = x_true.reshape(-1, 1, h, w)

        grad_x_rec = F.conv2d(x_rec_flat, self.kernel_x, padding=1)
        grad_y_rec = F.conv2d(x_rec_flat, self.kernel_y, padding=1)

        grad_x_true = F.conv2d(x_true_flat, self.kernel_x, padding=1)
        grad_y_true = F.conv2d(x_true_flat, self.kernel_y, padding=1)

        loss = F.l1_loss(grad_x_rec, grad_x_true) + F.l1_loss(grad_y_rec, grad_y_true)
        return loss


class DOFASemanticLoss(nn.Module):
    """Computes semantic feature loss using DOFA features."""

    def __init__(self, dofa_net: nn.Module):
        super().__init__()
        self.dofa_net = dofa_net
        # Freeze DOFA
        for p in self.dofa_net.parameters():
            p.requires_grad = False

    def forward(
        self, inputs: torch.Tensor, reconstructions: torch.Tensor, wvs: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            f_in = self.dofa_net.forward_features(inputs, wvs)
        # Allow gradients to flow to reconstructions
        f_rec = self.dofa_net.forward_features(reconstructions, wvs)

        l_feat = 0.0
        for fi, fr in zip(f_in, f_rec):
            l_feat += (1.0 - F.cosine_similarity(fi, fr, dim=1)).mean()

        return l_feat


class EOConsistencyLoss(nn.Module):
    def __init__(
        self,
        pixel_weight: float = 1.0,  # Basic L1 loss
        spectral_weight: float = 0.5,  # SAM for spectral accuracy
        spatial_weight: float = 0.5,  # Gradient for edges
        freq_weight: float = 0.1,  # FFT for textures
        feature_weight: float = 0.0,  # Optional DOFA features
        spectral_start_step: int = 1000,
        spatial_start_step: int = 2000,
        freq_start_step: int = 5000,
        feature_start_step: int = 5000,
        dofa_net: nn.Module = None,
    ):
        """Initializes the EO Consistency Loss with multiple components.

        Args:
            pixel_weight: Weight for pixel-wise L1 loss.
            spectral_weight: Weight for spectral angle mapper loss, useful for multispectral data.
            spatial_weight: Weight for spatial gradient loss to preserve edges.
            freq_weight: Weight for focal frequency loss to capture textures.
            feature_weight: Weight for semantic feature loss using a pretrained DOFA network.
            spectral_start_step: Global step to start applying spectral loss.
            spatial_start_step: Global step to start applying spatial loss.
            freq_start_step: Global step to start applying frequency loss.
            feature_start_step: Global step to start applying feature loss.
            dofa_net: Pretrained DOFA network for feature extraction (if feature_weight > 0).

        """
        super().__init__()

        self.starts = {
            'spectral': spectral_start_step,
            'spatial': spatial_start_step,
            'freq': freq_start_step,
            'feature': feature_start_step,
        }
        self.weights = {
            'pixel': pixel_weight,
            'spectral': spectral_weight,
            'spatial': spatial_weight,
            'freq': freq_weight,
            'feature': feature_weight,
        }

        self.sam_loss = SpectralAngleMapperLoss()
        self.grad_loss = SpatialGradientLoss()
        self.fft_loss = FFL(loss_weight=1.0, alpha=1.0)
        self.feature_loss = DOFASemanticLoss(dofa_net) if dofa_net is not None else None

    def forward(
        self,
        inputs: torch.Tensor,
        wvs: torch.Tensor,
        reconstructions: torch.Tensor,
        global_step: int = 0,
        split: str = 'train',
        **kwargs,
    ):
        logs = {}
        # Initialize as tensor to ensure device consistency
        total_loss = torch.tensor(0.0, device=inputs.device)

        # 1. Pixel Loss (Always Active)
        if self.weights['pixel'] > 0:
            l_pix = F.l1_loss(reconstructions, inputs)
            total_loss = total_loss + self.weights['pixel'] * l_pix
            logs[f'{split}/loss_rec'] = l_pix.detach()

        # 2. Spectral Loss (Scheduled)
        if self.weights['spectral'] > 0:
            if global_step >= self.starts['spectral']:
                l_sam = self.sam_loss(reconstructions, inputs)
                total_loss = total_loss + self.weights['spectral'] * l_sam
                logs[f'{split}/loss_spectral'] = l_sam.detach()
            else:
                # Log 0.0 so charts don't break during warm-up
                logs[f'{split}/loss_spectral'] = torch.tensor(0.0, device=inputs.device)

        # 3. Spatial Loss (Scheduled)
        if self.weights['spatial'] > 0:
            if global_step >= self.starts['spatial']:
                l_spat = self.grad_loss(reconstructions, inputs)
                total_loss = total_loss + self.weights['spatial'] * l_spat
                logs[f'{split}/loss_spatial'] = l_spat.detach()
            else:
                logs[f'{split}/loss_spatial'] = torch.tensor(0.0, device=inputs.device)

        # 4. Frequency Loss (Scheduled)
        if self.weights['freq'] > 0:
            if global_step >= self.starts['freq']:
                l_freq = self.fft_loss(reconstructions, inputs)
                total_loss = total_loss + self.weights['freq'] * l_freq
                logs[f'{split}/loss_freq'] = l_freq.detach()
            else:
                logs[f'{split}/loss_freq'] = torch.tensor(0.0, device=inputs.device)

        # 5. Semantic Feature Loss (Scheduled + Requires Net)
        if self.weights['feature'] > 0:
            if global_step >= self.starts['feature']:
                l_feat = self.feature_loss(inputs, reconstructions, wvs)
                total_loss += self.weights['feature'] * l_feat
                logs[f'{split}/loss_feature'] = l_feat.detach()
            else:
                logs[f'{split}/loss_feature'] = torch.tensor(0.0, device=inputs.device)

        logs[f'{split}/loss_total'] = total_loss.detach()

        return total_loss, logs
