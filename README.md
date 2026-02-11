# EO-VAE

Earth Observation Variational AutoEncoder (EO-VAE) repository. This project focuses on training Variational AutoEncoders for remote sensing data and leveraging the latent space for downstream tasks like super-resolution.

## Installation

```shell
pip install -e .
```

## Workflow

The project workflow is divided into two main parts: adapting the VAE to EO data, and training downstream generative models.

### Part 1: Training the AutoEncoder (VAE)

**Step 1.1: Weight Distillation**
First, distill the RGB weights from the flexible DOFA input to the RGB channels of the pretrained Flux2 AutoEncoder.
- **Script**: `weight_distill_train.py`
```shell
python weight_distill_train.py --config configs/eo-vae.yaml --teacher-ckpt <path_to_flux_ae>
```

**Step 1.2: Multi-Modal Finetuning**
Finetune the model on multiple modalities of TerraMesh data using the distilled weights.
- **Script**: `train.py`
```shell
python train.py --config configs/eo-vae.yaml --ckpt <path_to_distilled_ckpt>
```

**Step 1.3: VAE Evaluation**
Perform visual and metric evaluation of the trained VAE.
- **Visuals**: `visual_eval.py`
- **Metrics**: `eval_metric_recon.py`

### Part 2: Latent Space Super-Resolution

This stage uses the frozen trained VAE to encode latents and trains a diffusion model for super-resolution.

**Step 2.1: Pre-compute Latents**
Use the frozen trained VAE to encode the dataset into latents.
- **Script**: `encode_latents.py`
```shell
python encode_latents.py \
    --sen2naip_root <path_to_data> \
    --config <path_to_vae_config> \
    --ckpt <path_to_vae_ckpt> \
    --output_root <output_dir> \
    --use_spatial_norm
```

**Step 2.2: Train Latent Diffusion**
Train the latent space diffusion model.
- **Script**: `train_super_res.py`
```shell
python train_super_res.py --config configs_superres/eo_vae_latent.yaml
```

**Step 2.3: Evaluation**
Evaluate the trained diffusion model by decoding predictions back to pixel space using the frozen AutoEncoder.
- **Visuals**: `eval_viz_super_res_compare.py`
- **Metrics**: `eval_metric_super_res.py`

## Directory Structure

- `configs/`: Configuration files for VAE training.
- `configs_superres/`: Configuration files for Super-Resolution training.
- `eo_vae/`: Source code for models, datasets, and utilities.
- `train.py`: Training scripts for VAE.
- `encode_latents.py`: Script to encode data to latent space.
- `train_super_res.py`: Training script for Super-Resolution.

## License

Apache-2.0 License

## Citation
Add paper here
