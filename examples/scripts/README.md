## Scripts

We also provided a training script as example allowing you to train VAE models on well known benchmark data set (mnist, cifar10, celeba ...).
The script can be launched with the following commandline

```bash
python training.py --dataset mnist --model_name ae --model_config 'configs/ae_config.json' --training_config 'configs/base_training_config.json'
```

The folder structure should be as follows:
```bash
.
├── configs # the model & training config files (you can amend these files as desired or specify the location of yours in '--model_config' )
│   ├── ae_config.json
│   ├── base_training_config.json
│   ├── beta_vae_config.json
│   ├── hvae_config.json
│   ├── rhvae_config.json
│   ├── vae_config.json
│   └── vamp_config.json
├── data # the dataset with train_data.npz and eval_data.npz files
│   ├── celeba
│   │   ├── eval_data.npz
│   │   └── train_data.npz
│   ├── cifar10
│   │   ├── eval_data.npz
│   │   └── train_data.npz
│   └── mnist
│       ├── eval_data.npz
│       └── train_data.npz
├── my_models # trained models are saved here
│   ├── AE_training_2021-10-15_16-07-04 
│   └── RHVAE_training_2021-10-15_15-54-27
├── README.md
└── training.py
```

**Note** The data in the `train_data.npz` and `eval_data.npz` files must be loadable as follows

```python
train_data = np.load(os.path.join(PATH, f'data/{args.dataset}', 'train_data.npz'))['data']
eval_data = np.load(os.path.join(PATH, f'data/{args.dataset}', 'eval_data.npz'))['data']
```
where `train_data` and `eval_data` have now the shape (n_img x im_channel x height x width)

## Testing Pretrained Models

We also provide scripts to test pretrained models on the CelebA dataset. These scripts can load models from HuggingFace Hub or local paths and test reconstruction, generation, and interpolation capabilities.

### Testing VQ-VAE on CelebA

**Note**: Currently, there is no pretrained VQ-VAE model available on HuggingFace Hub. You'll need to train a model first before testing.

```bash
# First, train a VQ-VAE model (see training section below)
# Then test the locally trained model
python test_vqvae_celeba.py --model_path ./my_models/VQVAE_training_2024-01-01_12-00-00/final_model --output_dir results

# With custom data path
python test_vqvae_celeba.py --model_path ./my_models/VQVAE_training_2024-01-01_12-00-00/final_model --data_path /path/to/celeba --output_dir results

# Skip generation tests (faster)
python test_vqvae_celeba.py --model_path ./my_models/VQVAE_training_2024-01-01_12-00-00/final_model --skip_generation
```

The script will:
1. Load the pretrained VQ-VAE model (from HuggingFace Hub or local path)
2. Load CelebA dataset
3. Test reconstruction on evaluation samples
4. Test generation using NormalSampler, GaussianMixtureSampler, and MAFSampler
5. Test interpolation between image pairs
6. Save visualization results to the output directory

### Testing VAEGAN on CelebA

```bash
# Test a model from HuggingFace Hub
python test_vaegan_celeba.py --model_path clementchadebec/vaegan_celeba --output_dir results

# Test a locally trained model
python test_vaegan_celeba.py --model_path ./my_models/VAEGAN_training_2024-01-01_12-00-00/final_model --output_dir results

# With custom data path
python test_vaegan_celeba.py --model_path clementchadebec/vaegan_celeba --data_path /path/to/celeba --output_dir results

# Skip generation tests (faster)
python test_vaegan_celeba.py --model_path clementchadebec/vaegan_celeba --skip_generation
```

The script will:
1. Load the pretrained VAEGAN model (from HuggingFace Hub or local path)
2. Load CelebA dataset
3. Test reconstruction on evaluation samples
4. Test generation using NormalSampler and GaussianMixtureSampler
5. Test interpolation between image pairs
6. Save visualization results to the output directory

**Note**: If using HuggingFace Hub, ensure you have `huggingface_hub` installed:
```bash
pip install huggingface_hub
```

If the model is private, you may need to log in:
```bash
huggingface-cli login
```

## Training VQ-VAE on CelebA (CPU-Optimized)

For CPU-based training, use the provided CPU-optimized configuration:

```bash
# Using the shell script
./train_vqvae_celeba_cpu.sh

# Or directly with Python
python training.py \
    --dataset celeba \
    --model_name vqvae \
    --model_config configs/celeba/vqvae_config.json \
    --training_config configs/celeba/vqvae_training_config_cpu.json \
    --nn convnet
```

The CPU-optimized configuration uses:
- Smaller batch sizes (8 for train/eval)
- Fewer epochs (5)
- No CUDA (`no_cuda: true`)
- No multiprocessing workers (0) to avoid memory issues