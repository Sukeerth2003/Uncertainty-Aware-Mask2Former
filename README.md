# Uncertainty-Aware Mask2Former (Core Extensions)

This repository contains the core architectural extensions and custom loss functions developed for my Master's Defense project: **Uncertainty-Aware Mask2Former for Autonomous Vehicles**. 

Rather than duplicating the entire Mask2Former framework, this repository isolates the novel dual-uncertainty contributions designed to replace standard deterministic classification.

## Core Contributions (`/src`)
Standard perception models are fundamentally overconfident. To solve this, we replace the Mask2Former Softmax head with a real-time safety buffer:
* `evidential_head.py`: An Evidential Deep Learning (EDL) head that outputs raw evidence to parameterize a Dirichlet distribution, quantifying categorical doubt (Semantic Uncertainty).
* `spatial_head.py`: A variance-based spatial head that maps boundary hesitation to a Gaussian distribution (Spatial Uncertainty).
* `losses.py`: Custom mathematical formulations including a Gaussian Negative Log-Likelihood (NLL) to attenuate spatial regression errors, and an Evidential KL-Divergence penalty.

## Usage
These files are designed to be integrated directly into the official [Mask2Former codebase](https://github.com/facebookresearch/Mask2Former). 
1. Replace the standard linear classification head in the Transformer Decoder with the `EvidentialClassHead`.
2. Route the mask features through the `SpatialUncertaintyHead`.
3. Swap the standard Cross-Entropy loss with the composite functions provided in `losses.py`.

## Documentation
The complete mathematical derivations, zero-shot evaluation on the ACDC adverse-weather dataset, and the "Trust-PQ Paradox" analysis can be found in the `/docs` folder, which includes the full Master's Report and Defense Presentation.
