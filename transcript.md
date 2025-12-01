## Title Slide

This talk is about variational autoencoders and some of their most influential extensions: $\beta$-VAE, VQ-VAE, and VAEGAN.  
I’m Aleksei Krotov, presenting together with Muhammad Hamza Mubashir, and this work was carried out for EECE-7398 with Prof. Sarah Ostadabbas at Northeastern University.  
We treated this project as a small research study: we reimplemented several VAE architectures from the original papers, trained them under a unified pipeline on CelebA, and then compared both their internal dynamics and their final image quality.

## Outline

I’ll start with the motivation and problem statement: why VAEs, and why these particular variants.  
Then I’ll review the core VAE formulation and introduce the three extensions—$\beta$-VAE, VQ-VAE, and VAEGAN—emphasizing how each one modifies the objective or the latent space.  
Next, I’ll describe our dataset and implementation details: how we train all models on CelebA using Pythae and the Northeastern Explorer cluster.  
After that, we move to qualitative results: reconstructions and generations for $\beta$-VAE, followed by side‑by‑side comparisons for VQ‑VAE and VAEGAN.  
We’ll then look at quantitative metrics, and finally I’ll close with a comparative discussion and key takeaways.

## Problem and Goal

The central question we ask is: **how do different VAE‑based architectures trade off reconstruction fidelity, disentanglement, and perceptual realism** on a realistic image dataset.  
CelebA, with its diverse faces and attributes, is a good setting because it is much richer than MNIST, yet still standard in the generative-modeling literature.  
Our goal is to **fairly compare** four models—VAE, $\beta$‑VAE, VQ‑VAE, and VAEGAN—by keeping as much as possible fixed: the dataset, the convolutional backbone, the optimizer family, and the evaluation metrics.  
We also go beyond pure likelihood proxies by including perceptual metrics (LPIPS, SSIM, GMSD), so that we can talk about **how the samples look** as well as how they score under the training objective.

## VAE Architecture

Let me briefly recap the standard VAE as introduced by Kingma & Welling.  
We assume a simple prior $p(z)=\mathcal{N}(0,I)$ over latent variables $z$, and we learn an encoder $q_\phi(z\mid x)$ and decoder $p_\theta(x\mid z)$.  
The encoder outputs a mean and log‑variance for a Gaussian in latent space, and we use the reparameterization trick—$z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0,I)$—to make sampling differentiable.  
Training maximizes the ELBO:
\[
  \mathcal{L}_{\text{VAE}} =
  \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
  - \mathrm{KL}(q_\phi(z \mid x)\,\|\,p(z)),
\]
so there is a **reconstruction term** and a **KL regularizer** that keeps the approximate posterior close to the prior.  
In our experiments, all models share the same convolutional encoder/decoder backbone from the Pythae framework so that architectural differences in the latent space and objective, not the backbone, drive the comparisons.

## $\beta$‑VAE and VQ‑VAE

The first extension is $\beta$‑VAE from Higgins et al.  
It keeps the same model but rescales the KL term:
\[
  \mathcal{L}_{\beta\text{-VAE}}
  = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
  - \beta\,\mathrm{KL}(q_\phi(z \mid x)\,\|\,p(z)).
\]
When $\beta > 1$, the KL penalty is stronger, which effectively limits the information the encoder is allowed to send through $z$.  
From an information‑bottleneck perspective, this pushes the model toward **disentangled, factorized latents**, but the price is that reconstructions become blurrier because each image can use fewer “nats” of information.

VQ‑VAE, from van den Oord et al., changes the latent space itself.  
Instead of sampling from a Gaussian, the encoder outputs continuous embeddings which are then **quantized** to the nearest vector in a learned codebook.  
The latent is now a grid of discrete indices, and the loss replaces the KL term with a vector‑quantization loss:
- a **commitment loss** that keeps encoder outputs close to their selected codebook vectors, and  
- a **codebook update term** that pulls the codebook vectors toward the encoder outputs (EMA in the original paper).  
This design gives a **discrete latent space** that behaves more like a learned dictionary; it is particularly powerful for hierarchical or autoregressive priors, but even on its own we can examine how well it reconstructs and generates CelebA faces.

## VAEGAN Architecture

The third variant is VAEGAN, following Larsen et al.  
Here, a VAE‑style encoder and decoder are coupled with a GAN discriminator $D(x)$, and the decoder plays the role of a GAN generator.  
Two key ideas from the paper are:
- **Feature‑space reconstruction**: instead of penalizing pixel‑wise errors only, VAEGAN measures an $L_2$ distance between discriminator feature maps of $x$ and $\hat{x}$ at an intermediate layer.  
- **Adversarial training**: the decoder also receives gradients from the discriminator’s GAN loss, encouraging generated samples to live on the manifold of realistic faces.  
The total loss decomposes into an encoder loss (KL + part of the feature loss), a decoder loss (feature loss + GAN generator loss), and a discriminator loss.  
This combination tends to produce **very sharp and realistic images**, but training is more delicate and the latent space is generally less interpretable than in $\beta$‑VAE.

## Dataset and Implementation

All experiments are conducted on **CelebA**, using the aligned and cropped $64\times 64$ face images.  
We work with roughly 162k training images and about 20k evaluation images; we pre‑process them into NumPy `.npz` files so that training and evaluation can use exactly the same data splits.  
All four models—VAE, $\beta$‑VAE, VQ‑VAE, and VAEGAN—are implemented through Pythae, which lets us plug different model configs into a shared training pipeline (`TrainingPipeline` and trainer configs).  
We run the experiments on the Northeastern **Explorer GPU cluster** using Nvidia **Tesla P100** GPUs, with configuration and reproducibility handled via JSON config files and shell scripts.  
Weights \& Biases is used for experiment tracking, and the full code, configs, and environment details are available in our GitHub repository, so every result in this talk is reproducible.

## Experiment Tracking and wandb Links

This slide emphasizes that our experiments are **fully inspectable**.  
For training, we log VQ‑VAE runs to a dedicated `vqvae-celeba` project and group the VAE and VAEGAN training runs in `vae-search-celeba`.  
These dashboards contain epoch‑wise loss curves, model configs, and hardware information, making it easy to check whether, for example, VAEGAN has stabilized or is still in an adversarial tug‑of‑war.  

For evaluation, we create separate wandb runs `vqvae-evaluation` and `vaegan-evaluation`, where we log detailed reconstruction and generation metrics; the complete tables are visible in the logs section.  
On the slide, we show three representative loss curves—one for the $\beta$‑VAE baseline, one for VQ‑VAE, and one for VAEGAN.  
You can see that **all models converge**, but VAEGAN in particular shows the characteristic oscillations of GAN training before settling into a good equilibrium.

## Visual Evaluation – $\beta$‑VAE: Reconstruction and Generation

This slide combines two key figures for the baseline VAE and $\beta$‑VAE.  
The upper image shows **reconstructions**: the top row has original CelebA faces, and the rows below show reconstructions for different $\beta$ values.  
As we move from $\beta = 0.5$ to $\beta = 8.0$, the faces become progressively blurrier: fine‑grained details like hair strands, glasses, and background texture are washed out.  
This is exactly what the $\beta$‑VAE paper describes: stronger KL pressure shrinks the information channel, forcing the model to keep only the coarsest structure of the image.

The second image on the slide shows **unconditional generations** for the same $\beta$ settings.  
We sample $z \sim \mathcal{N}(0,I)$ and decode to images.  
At low $\beta$, samples are diverse and reasonably sharp; at high $\beta$, they remain recognizable faces but lose individuality and appear almost “averaged out.”  
The combined takeaway is that increasing $\beta$ systematically trades away both reconstruction fidelity and sample sharpness in exchange for a more constrained latent representation.

## Visual Evaluation – VQ‑VAE and VAEGAN

Now we turn to VQ‑VAE and VAEGAN, using a standardized visualization: three rows labeled **Original, Reconstruction, Generated**.  
For VQ‑VAE, notice that reconstructions preserve identity, pose, and coarse hairstyle quite well; the discrete codebook is still expressive enough to capture most facial structure.  
The generated row shows samples obtained by sampling from the learned codebook and decoding—these are coherent faces, though slightly softer than VAEGAN’s, reflecting the fact that the loss is still dominated by mean‑squared error.  

For VAEGAN, the pattern changes.  
Reconstructions remain close to the originals in terms of high‑level content, but they are often **sharper and more contrasty** than the VAE or VQ‑VAE outputs because the model is matching discriminator features instead of pixels.  
The generated faces in the bottom row look very realistic: clear eyes, well‑defined hair, and plausible lighting.  
Even without numbers, these grids visually support the idea that VAEGAN is optimized for perceptual similarity rather than exact pixel match.

## Numerical Evaluation – Reconstruction Metrics

The next slide presents reconstruction metrics for all models.  
The table reports BCE, MSE, KL, ELBO, and three perceptual scores: LPIPS (lower is better), SSIM (higher is better), and GMSD (lower is better).  

For the $\beta$‑VAE rows, increasing $\beta$ from $0.5$ to $4$ drives the KL term down from roughly $46$ to about $1$, which means the posterior is being forced very close to the prior.  
At the same time, reconstruction MSE and LPIPS increase, and SSIM decreases—quantitatively confirming the blur we saw in the images.  

VQ‑VAE achieves an MSE of about $0.007$ and SSIM around $0.69$, with LPIPS $\approx 0.26$ and GMSD $\approx 0.155$.  
These are competitive with the best $\beta$‑VAE setting, despite using a discrete codebook and a VQ loss instead of KL.  
VAEGAN’s reconstruction numbers (MSE $\approx 0.009$, BCE $\approx 0.51$) should be interpreted cautiously: because it is trained to match discriminator features, not pixels, pixel‑space errors are not the primary objective, but we still report them for completeness.

## Numerical Evaluation – Generation Metrics

The second table focuses on **generation and self‑reconstruction** metrics.  
For $\beta$‑VAE, moderate $\beta$ again provides a compromise: at $\beta = 0.5$, MSE is about $0.006$, SSIM about $0.87$, and LPIPS around $0.093$.  
As $\beta$ increases, these numbers drift in the direction of worse reconstruction—slightly higher MSE and LPIPS, and lower SSIM—matching our qualitative impression.

For VQ‑VAE, sampling from the codebook and then reconstructing yields MSE $\approx 0.009$, LPIPS $\approx 0.154$, SSIM $\approx 0.80$, and GMSD $\approx 0.113$.  
This shows that even when we generate from discrete codes, VQ‑VAE maintains strong self‑consistency: generated faces can be passed back through the model without dramatic degradation.  

VAEGAN clearly stands out: its generation MSE is about $0.001$, LPIPS drops to roughly $0.033$, SSIM jumps to around $0.97$, and GMSD falls to about $0.031$.  
These metrics capture what we saw in the visual grids—the model produces highly realistic, perceptually close self‑reconstructions, which is exactly what a feature‑space, adversarial objective is designed to encourage.

## Discussion – Effects of $\beta$

Putting these observations together, we can revisit the role of $\beta$.  
Mathematically, scaling the KL term by $\beta$ changes the **effective channel capacity** between $x$ and $z$: higher $\beta$ means the model is punished more for deviating from the prior, so it encodes fewer bits about each image.  
This tends to move the aggregated posterior toward the factorized prior, which is helpful for disentanglement and for learning a smoother latent space.  
However, on a rich dataset like CelebA, we see the cost very clearly: reconstructions and generations both lose fine detail, and the perceptual metrics degrade.  
So $\beta$‑VAE behaves exactly as the original paper predicts, but the “sweet spot” for usable image quality depends heavily on how much disentanglement one is willing to sacrifice.

## Discussion – VQ‑VAE Effects

For VQ‑VAE, the main story is that we obtain a **discrete, structured latent space** without sacrificing much reconstruction quality.  
The codebook entries can be thought of as “prototype patches” in latent space; during training, the commitment and codebook losses ensure that encoder outputs stay near these prototypes and that the prototypes move toward frequently used regions.  
The result is that similar faces tend to reuse similar codes, which can be exploited for compression or for downstream tasks like clustering and semantic editing.  
Our metrics show that both reconstruction and generation/self‑reconstruction performance are close to the best $\beta$‑VAE baseline, so in this setting VQ‑VAE is a very competitive alternative to continuous Gaussian latents.

## Discussion – VAEGAN and Comparative View

VAEGAN is clearly the **perceptual winner** in our experiments.  
By combining a KL term, feature‑space reconstruction loss, and an adversarial loss, it encourages the decoder to produce samples that not only resemble the data distribution in a global sense but also align with human perception of sharpness and realism.  
The trade‑offs are that the latent space is no longer as cleanly interpretable as in $\beta$‑VAE, and the optimization is more fragile because of the GAN component and coupled optimizers for encoder, decoder, and discriminator.  

Taken together, the three variants illustrate different design philosophies:
- $\beta$‑VAE: **regularize harder for disentanglement**, accept worse pixel‑level quality.  
- VQ‑VAE: **move to discrete latents** and codebook structure while keeping strong reconstructions.  
- VAEGAN: **optimize for perceptual realism**, embracing adversarial training and feature‑space losses.

## Summary and Future Work

To summarize, we implemented and evaluated VAE, $\beta$‑VAE, VQ‑VAE, and VAEGAN on CelebA using a unified Pythae‑based pipeline, controlled configs, and common metrics.  
Our results reproduce the classic $\beta$‑VAE trade‑off, show that VQ‑VAE can match strong baselines while offering discrete structure, and demonstrate that VAEGAN delivers the best perceptual metrics and the sharpest, most realistic samples.  
Beyond the numbers, this project gave us hands‑on experience with the internal dynamics of each model—how the losses behave, how training curves look in practice, and how architectural choices change the character of the learned representations.  
For future work, we’d like to scale up the architectures, apply formal disentanglement metrics on CelebA, and explore how these learned latents transfer to tasks such as attribute prediction or controllable face editing.  
Thank you, and I’m happy to take questions.

