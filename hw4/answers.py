r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
import torch

# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.5, learn_rate=1e-3, eps=1e-8, num_workers=2,
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=1.0,
        delta=1.0,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['z_dim'] = hypers['h_dim'] = 256
    hypers['betas'] = (0.95, 0.95)
    hypers['x_sigma2'] = 0.00084
    hypers['learn_rate'] = 0.00186
    # ========================
    return hypers

# TODO: fill answers
part2_q1 = r"""
**Your answer:**

The $\sigma^2$ determines the level of random in the model.
We can view it as the weight that will determine data-reconstruction loss.
Low values will make low variability to the data which the model creates 
so it will be closer to the dataset.
High values will make high variability to the data which the model creates
so it will be further away from the dataset.

"""

part2_q2 = r"""
**Your answer:**

1. KL divergence loss purpose is regularization, so the encoder output
distribution will be similar to standard normal distribution.
Reconstruction loss purpose is for that the images which are the model output
will be similar to the original images. 

2. The latent-space distribution is affected by the KL loss term by that the
encoder output distribution will be close to normal distribution.

3. The benefit of this effect is that because the distribution will be close
to a normal distribution at the middle of the latent space which will make the decoder
to decode the data generated randomly from the like normal distribution latent space
so that points which are neighbors will be alike.  

"""

part2_q3 = r"""
**Your answer:**

We maximize the evidence distribution at the start to make the model generated 
samples to be close to the original samples.

"""

part2_q4 = r"""
**Your answer:**

We model the log of the latent-space variance corresponding to an input 
to achieve stability when we map $[0,1]$ to $[-\infty, log(1)]$.
This is instead of modeling the variance directly.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 4 answers
# ==============


def part4_affine_backward(ctx, grad_output):
    # ====== YOUR CODE: ======
    # TODO: implement
    raise NotImplementedError()
    # ========================
