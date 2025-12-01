## Title Slide

This talk is about variational autoencoders and some of their most influential extensions: $\beta$-VAE, VQ-VAE, and VAEGAN.  
I’m Aleksei Krotov, presenting together with Muhammad Hamza Mubashir, and this work was carried out for EECE-7398 with Prof. Sarah Ostadabbas at Northeastern University.  
Our focus is on understanding how these architectures differ in their objectives, what kinds of representations they learn on the CelebA dataset, and how those differences show up in both quantitative metrics and visual quality.

## Outline

I’ll start with the motivation and problem statement: why VAEs, and why compare these specific variants.  
Then I’ll review the core VAE formulation and introduce the three extensions: $\beta$-VAE, VQ-VAE, and VAEGAN, highlighting what each one changes in the architecture or objective.  
Next, I’ll describe the dataset and implementation details—how we trained these models on CelebA using the Pythae framework and our HPC setup.  
After that, we’ll look at qualitative results: reconstructions and generations for the different models.  
Then I’ll present quantitative results—both classical VAE metrics like ELBO and newer perceptual metrics like LPIPS, SSIM, and GMSD.  
Finally, I’ll wrap up with a comparative discussion, key takeaways, and directions for future work.

## Problem and Goal

The core problem we’re addressing is how different VAE-based architectures trade off reconstruction quality, disentanglement, and sample realism on a challenging, real-world image dataset.  
CelebA, with its diverse face images and many attributes, is a good testbed because it’s more complex than MNIST-style benchmarks yet still widely used in generative modeling.  
Our goal is to implement and fairly compare four models—VAE, $\beta$-VAE, VQ-VAE, and VAEGAN—using a shared training pipeline, the same dataset, and a consistent set of evaluation metrics.  
We also deliberately include modern perceptual metrics like LPIPS, alongside SSIM and GMSD, so that we can talk not only about likelihood-style objectives but also about how “good” the images look.  
By controlling for implementation details, we want differences in performance to primarily reflect differences in model design rather than differences in training code or hyperparameters.

## VAE Architecture

Let me briefly recap the standard VAE formulation.  
In a VAE, we assume a prior $p(z) = \mathcal{N}(0, I)$ over latent variables $z$, and we learn a probabilistic encoder $q_\phi(z \mid x)$ and decoder $p_\theta(x \mid z)$.  
The encoder outputs a mean and variance for a Gaussian in latent space, and we use the reparameterization trick to sample $z$ in a differentiable way.  
Training maximizes the evidence lower bound, or ELBO: an expected reconstruction term plus a KL divergence term that pushes the approximate posterior $q_\phi(z \mid x)$ towards the prior.  
In our experiments, we use convolutional encoders and decoders implemented in the Pythae library, which gives us a consistent backbone across all models.

## $\beta$-VAE and VQ-VAE

Now let’s look at two important extensions: $\beta$-VAE and VQ-VAE.  
$\beta$-VAE keeps the same generative story as a standard VAE, but in the objective it multiplies the KL term by a factor $\beta > 1$.  
Intuitively, this increases the pressure for the posterior to match the prior, which tends to encourage more factorized and potentially more disentangled latent representations.  
The downside is that as $\beta$ grows, the model is allowed to encode less information about each individual image, which usually hurts reconstruction quality and sometimes sample diversity.  

VQ-VAE takes a different route: instead of a continuous Gaussian latent, it uses a learned discrete codebook.  
The encoder outputs continuous embeddings, which are then quantized to the nearest codebook vector; those discrete codes are what the decoder sees.  
The loss replaces the KL term with a vector-quantization loss that has a commitment component and a codebook-update component.  
This gives us a discrete latent space with a fixed number of codes, which can be very powerful for downstream tasks and for building hierarchical generative models.

## VAEGAN Architecture

The third extension we consider is VAEGAN.  
VAEGAN combines a VAE-style encoder and decoder with a GAN-style discriminator, so the decoder effectively acts as a generator in an adversarial setting.  
Instead of measuring reconstruction purely in pixel space, VAEGAN computes a reconstruction loss in the feature space of the discriminator—comparing intermediate feature maps for real and reconstructed images.  
This feature-space reconstruction tends to correlate better with human perception, leading to sharper and more realistic images than pure pixel-wise MSE or BCE.  
On top of that, the discriminator also provides an adversarial loss term that pushes generated images towards the distribution of real images.  
The total loss splits into encoder, decoder, and discriminator components, with the decoder loss balancing feature-space reconstruction against adversarial feedback.

## Dataset and Implementation

All experiments are conducted on the CelebA dataset, using the aligned and cropped $64\times64$ face images.  
We work with roughly 162 thousand training images and about 20 thousand evaluation images, stored in preprocessed NumPy arrays for efficient loading.  
Training is implemented using the Pythae library, which provides consistent implementations of VAE, $\beta$-VAE, VQ-VAE, and VAEGAN, all with convolutional backbones.  
We run our experiments on the Northeastern Explorer GPU cluster, using modern GPUs such as A100 and V100, and manage the environment with conda.  
Weights \& Biases is used for experiment tracking, so every run has full logs of losses, metrics, and configurations, and the entire code and configuration setup is available in our GitHub repository.

## Experiment Tracking and wandb Links

On this slide, I want to highlight how we exposed our experiments so that others can inspect them in detail.  
All of our key training and evaluation runs are logged on Weights \& Biases (wandb), and we provide direct links for anyone who wants to explore the training dynamics or metric histories.  

For training, we have a dedicated project for VQ-VAE on CelebA, available at the \emph{vqvae-celeba} dashboard, and another project that aggregates the VAE and VAEGAN training runs on CelebA, under \emph{vae-search-celeba}  
These dashboards show epoch-wise loss curves, learning rate schedules, and other training diagnostics.  

For evaluation, we created separate wandb runs for VQ-VAE and VAEGAN—\emph{vqvae-evaluation} and \emph{vaegan-evaluation}.  
If someone wants to dig into the precise reconstruction and generation metrics we report, they can view the full metric tables in the logs section of those evaluation runs.  

On the slide, we also show three representative training curves: one for the baseline/BetaVAE loss, one for the VQ-VAE epoch loss, and one for the VAEGAN epoch loss.  
These plots illustrate that all models reach a relatively stable regime, but they do so with different convergence behaviors, partly reflecting their different objective functions and optimization difficulties.

## Visual Evaluation – Reconstruction (1)

Here we show an example reconstruction comparison focusing on the baseline VAE and $\beta$-VAE.  
The top row in the figure contains original CelebA faces, and the subsequent rows show reconstructions for different models.  
Even without numbers, you can see that as we increase $\beta$, reconstructions become blurrier and less faithful to fine details like hair texture, facial wrinkles, or accessories.  
This visual degradation matches what we expect: pushing the KL term harder means the model can’t afford to devote as much capacity to each individual example.  
The key point here is that $\beta$-VAE is explicitly sacrificing reconstruction fidelity in order to buy a more regularized, potentially more disentangled latent space.

## Visual Evaluation – Reconstruction (2)

On this slide, we zoom out and compare reconstructions across all the models we trained: VAE, several $\beta$-VAE settings, VQ-VAE, and VAEGAN.  
The VAE and small-$\beta$ models generally give the sharpest pixel-level reconstructions but may encode entangled latent factors.  
VQ-VAE reconstructions are competitive: they capture high-level structure and identity fairly well, with only mild blurring relative to the best VAE settings.  
VAEGAN’s reconstructions often look visually sharp and realistic, but because it matches discriminator features rather than pixels, the reconstructed image may deviate slightly from the exact input at a pixel level.  
Overall, visually, you can see the trade-offs: some models preserve more exact detail, while others favor more perceptual realism or more structured latent representations.

## Visual Evaluation – Generation

Here we compare unconditional generations from the models, especially focusing on VAE versus $\beta$-VAE.  
We sample from the prior in latent space and decode to images, then visually inspect the quality and diversity.  
For the baseline VAE, samples are usually plausible faces but can suffer from softness and occasionally artifacts, especially in backgrounds or hair.  
As we increase $\beta$, sample diversity can remain reasonable, but individual samples become blurrier, reflecting the lower information capacity in the latent code.  
Later in the discussion, we’ll compare these generations against what we obtain from VQ-VAE and VAEGAN, which tend to generate crisper and more realistic samples.

## Numerical Evaluation – Reconstruction Metrics

Let’s move to the quantitative results, starting with reconstruction metrics.  
The first table summarizes reconstruction performance in terms of BCE, MSE, KL, ELBO, and the perceptual metrics LPIPS, SSIM, and GMSD.  
For the $\beta$-VAE rows, we see that as $\beta$ increases from 0.5 to 4, KL drops dramatically—from roughly 46 down to about 1—while MSE and LPIPS increase and SSIM decreases.  
This numerically confirms what we saw visually: higher $\beta$ means more regularization and worse reconstructions.  

For VQ-VAE, we report an MSE of about 0.007, LPIPS around 0.26, and SSIM around 0.69, with a GMSD of about 0.155.  
These numbers are competitive with the best $\beta$-VAE settings, despite VQ-VAE using a discrete codebook and a VQ loss rather than a KL term.  
For VAEGAN, reconstruction MSE is about 0.009, BCE around 0.51, KL about 162.6, ELBO around 162.6 as well, with LPIPS near 0.30, SSIM around 0.66, and GMSD about 0.16.  
Keep in mind that for VAEGAN, pixel-level recon errors don’t fully capture quality, because the model is optimized in feature space; we’ll see its strength more clearly in the generation metrics.

## Numerical Evaluation – Generation Metrics

The second table reports generation and self-reconstruction metrics.  
For the $\beta$-VAE baselines, we again see that moderate $\beta$ gives a good balance: for example, at $\beta = 0.5$ MSE is around 0.006, SSIM around 0.87, and LPIPS about 0.093.  
As $\beta$ increases, MSE and LPIPS increase slightly and SSIM decreases, consistent with the idea that higher $\beta$ trades reconstruction quality for regularization.  

For VQ-VAE, the generation/self-reconstruction MSE is roughly 0.009, LPIPS about 0.154, SSIM around 0.80, and GMSD about 0.113.  
This shows that when we sample from the codebook and reconstruct, VQ-VAE maintains reasonably good fidelity and structure, even though its latent space is discrete.  

VAEGAN stands out in these metrics: its generation MSE is very low, about 0.001, LPIPS drops to roughly 0.033, SSIM jumps to around 0.97, and GMSD falls to about 0.031.  
These are extremely strong perceptual metrics, indicating that VAEGAN’s generations are both structurally and perceptually close to their self-reconstructions, and that the model produces high-quality samples.

## Discussion – Effects of $\beta$

Let’s interpret these results, starting with the effect of $\beta$.  
The KL term goes from roughly 46 at $\beta = 0.5$ down to about 1 at $\beta = 4$, which means the approximate posterior is being pushed much closer to the prior as $\beta$ grows.  
This is exactly what we want for disentanglement: latents are encouraged to align with the factorized prior, which can separate underlying factors of variation.  
However, the cost is clear in both MSE and perceptual metrics: reconstructions become blurrier, LPIPS increases, and SSIM drops.  
On CelebA, we didn’t fully visualize classic disentangled factors like pose or lighting as cleanly as in synthetic datasets like dSprites, but the numerical trends closely echo the original $\beta$-VAE observations.

## Discussion – VQ-VAE Effects

For VQ-VAE, the key story is that we get competitive reconstruction quality while moving to a discrete latent space.  
An MSE around 0.007 and SSIM close to 0.69 indicate that reconstructions are in the same ballpark as the best $\beta$ settings, and LPIPS and GMSD show that the perceptual quality is respectable.  
On the generation side, VQ-VAE maintains similar performance, with MSE around 0.009 and SSIM around 0.80, suggesting that sampling from the codebook does not severely degrade quality.  
The interesting aspect is the structure of the latent space: codebook entries are reused, which can encourage clustering of similar images and opens the door to discrete latent manipulations.  
So VQ-VAE offers a middle ground: better-structured latents and strong reconstructions, without the pixel-level sharpness that we’ll see from VAEGAN but with significantly more structure than a plain VAE.

## Discussion – VAEGAN and Comparative View

VAEGAN is the clear winner in terms of perceptual generation quality.  
With LPIPS around 0.033 and SSIM around 0.97 on generated and self-reconstructed samples, it produces visually very sharp and realistic faces.  
These numbers, and the corresponding images, show that optimizing feature-space reconstruction plus an adversarial objective helps the model capture high-level semantics and fine details.  
However, this comes with some trade-offs: the latent space is less easily interpretable than in $\beta$-VAE, and training is more delicate due to the adversarial component and coupled optimizers.  
Putting everything together, we can say: $\beta$-VAE emphasizes disentanglement, VQ-VAE emphasizes discrete latent structure with good reconstructions, and VAEGAN emphasizes perceptual realism and sample quality.

## Summary and Future Work

To summarize, we implemented and evaluated four VAE-family models—VAE, $\beta$-VAE, VQ-VAE, and VAEGAN—on the CelebA dataset using a unified Pythae-based pipeline.  
Our results confirm the classic $\beta$-VAE trade-off: increasing $\beta$ improves KL regularization and potential disentanglement, but degrades reconstruction and generation quality.  
VQ-VAE shows that moving to a discrete codebook can preserve strong reconstructions while giving us a more structured latent space that may be better suited for downstream tasks.  
VAEGAN, with its feature-space reconstruction and adversarial training, achieves the best perceptual metrics and the sharpest, most realistic samples.  
For future work, we’d like to scale up the architectures, explore more rigorous disentanglement metrics on CelebA, and study how these learned representations transfer to tasks like attribute prediction or editing.  
Thank you, and I’m happy to take questions.



