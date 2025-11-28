# Information about Pretrained Models

## How to Access Pretrained Models

According to the main README.md, pretrained models can be accessed using:

```python
from pythae.models import AutoModel
my_downloaded_vae = AutoModel.load_from_hf_hub(hf_hub_path="path_to_hf_repo")
```

## Available Pretrained Models

The list of available pretrained models is documented in the **reproducibility section**:
- Location: `examples/scripts/reproducibility/README.md`
- Table format showing: Model, Dataset, Metric, Obtained value, Reference value, Reference paper/code, and **Trained model link**

### Currently Available Models (from reproducibility table):

1. **VAE** - Binary MNIST - [link](https://huggingface.co/clementchadebec/reproduced_vae)
2. **VAMP** (K=500) - Binary MNIST - [link](https://huggingface.co/clementchadebec/reproduced_vamp)
3. **SVAE** - Dyn. Binarized MNIST - [link](https://huggingface.co/clementchadebec/reproduced_svae)
4. **PoincareVAE (Wrapped)** - MNIST - [link](https://huggingface.co/clementchadebec/reproduced_wrapped_poincare_vae)
5. **IWAE** (n_samples=50) - Binary MNIST - [link](https://huggingface.co/clementchadebec/reproduced_iwae)
6. **MIWAE** (M=8, K=8) - Dyn. Binarized MNIST - [link](https://huggingface.co/clementchadebec/reproduced_miwae)
7. **PIWAE** (M=8, K=8) - Dyn. Binarized MNIST - [link](https://huggingface.co/clementchadebec/reproduced_piwae)
8. **CIWAE** (beta=0.05) - Dyn. Binarized MNIST - [link](https://huggingface.co/clementchadebec/reproduced_ciwae)
9. **HVAE** (n_lf=4) - Binary MNIST - [link](https://huggingface.co/clementchadebec/reproduced_hvae)
10. **BetaTCVAE** - DSPRITES - [link](https://huggingface.co/clementchadebec/reproduced_beta_tc_vae)
11. **RAE_L2** - MNIST - [link](https://huggingface.co/clementchadebec/reproduced_rae_l2)
12. **RAE_GP** - MNIST - [link](https://huggingface.co/clementchadebec/reproduced_rae_gp)
13. **WAE** - CELEBA 64 - [link](https://huggingface.co/clementchadebec/reproduced_wae)
14. **AAE** - CELEBA 64 - [link](https://huggingface.co/clementchadebec/reproduced_aae)

## Models NOT Available as Pretrained

Based on the reproducibility table, the following models **do NOT** have pretrained versions available:

- ❌ **VQ-VAE** (Vector Quantized VAE)
- ❌ **VAEGAN** (Variational Autoencoder GAN)
- ❌ **BetaVAE**
- ❌ **FactorVAE**
- ❌ **DisentangledBetaVAE**
- ❌ **VAE_LinNF**
- ❌ **VAE_IAF**
- ❌ **MSSSIM_VAE**
- ❌ **INFOVAE_MMD**
- ❌ **RHVAE**
- ❌ **HRQVAE**
- ❌ **HVAE** (on CelebA)

## Important Notes

1. **No CelebA pretrained models for VQ-VAE or VAEGAN**: Neither VQ-VAE nor VAEGAN have pretrained models available on HuggingFace Hub, especially not for CelebA dataset.

2. **Training required**: For VQ-VAE and VAEGAN on CelebA, you must train the models yourself using the provided training scripts.

3. **How to check for models**: 
   - Check the reproducibility README for official pretrained models
   - Search HuggingFace Hub directly: `https://huggingface.co/models?search=clementchadebec`
   - Try loading: `AutoModel.load_from_hf_hub("clementchadebec/model_name")` - will raise error if not found

4. **Available CelebA models**: Only **WAE** and **AAE** have pretrained models available for CelebA dataset.

## Conclusion

For VQ-VAE and VAEGAN on CelebA, you need to:
1. Train the models yourself using the provided training scripts
2. Use the CPU-optimized configurations we've created
3. After training, you can test the models using the test scripts we've prepared

