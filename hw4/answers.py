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
    # TODO: Tweak the hyperparameters if needed. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
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
    # TODO: Tweak the hyperparameters.  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #  You can also add new ones if you need them for your model implementation.
    # ====== YOUR CODE: ======
    # ========================
    return hp


part1_q1 = r"""
If we reduce a baseline in the policy gradient our variance will be reduced since
some values that were previously positive and were increasing our variance will now
become negative after the reduction,(only values that were bigger than the 
mean will stay positive after the reduction). This way the variance is reduced 
as the policy ignores all trajectories with negative values.

**Example**: A model where there is more than a single trajectory. 
As explained above, as we Reduce a baseline, some of the trajectories will become negative
have a positive value for the reward and thus fewer trajectories will be considered.
"""


part1_q2 = r"""
State value indicate how good a given state is for an agent.
In our AAC, when we use estimated q-values (indicate how good an action is in a given state compared to the rest of 
the actions.), the model repeatedly learns state advantages, so when a new action is chosen, it is determined not only
by its q value, but also by how much it can improve the state values. This validates our approximation, and allows us
 with each approximation, to get better results since we gain lower variance with each step.
"""


part1_q3 = r"""
                                    **1**
                                    
                                <loss_p>
Negative loss of epg and vpg start at -90 and increases up to 0 during the episodes. No baseline reduction used.
For bpg and 'cpg' we can see changes around 0 in the loss, due to the use of baseline reduction.


                                <baseline>
The graphs contain the only methods that use baseline reduction: cpg and bpg.
the other methods are not in this graph. Both methods begin with a baseline 
of around -60 which rises towards 10 as the episodes progress. Bigger 
baseline values assist the network in increasing the reward.


                                <loss_e>
The epg and cpg models start at -0.5 entropy, which goes up to 0 as we run the episodes. A uniform
probability distribution is more likely here the higher absolute values we get. 
On the other hand, lower absolute values tend to point to a convergence of our network to an improved
policy. 

We can see that the best choice is cpg since we use baseline subtraction in it, which makes the model converge faster
then epg.

Note that the graphs only shows the models that use entropy loss, the other models 
are not in this graph.


                                <mean_reward>
While all methods start with a reward of -200 and rise 
as the episodes progress, there is a difference in how fast each method's 
reward rises. As can be seen in the graph, the methods which use a baseline 
rise a lot faster and higher than those which don't use a baseline. 




                                    **2**

                                <loss_p>
aac is better than vpg and epg and so it will approximate the state value better then other two.

This is inferred from the fact that all models start with a similar policy loss, but aac surpasses vpg 
and epg quickly, which means it has a lower trajectory loss.

                                <baseline>
The baseline graphs does not change since aac does not use baseline reduction.

                                <loss_e> 
aac has a lower entropy loss and diminishes faster than the entropy loss of epg and cpg in absolute value,
and so we would prefer to use aac.

                                <mean_reward>
aac yields better results in the mean reward as well, as we can see that it reaches 130.
Although in the first ~1k episodes aac performed worse and even decreased, in episodes post 1k
It outperformed all other methods substantially, ending up by episode
1200 with a much higher mean reward than the other candidates.
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
    # TODO: Tweak the hyperparameters to train your GAN. !!!!!!!!!!!!!!!!!
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 16
    hypers['z_dim'] = 256
    hypers['label_noise'] = 0.35
    hypers['data_label'] = 1
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['lr'] = 0.0005
    hypers['discriminator_optimizer']['betas'] = (0.5, 0.99)
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0004
    hypers['generator_optimizer']['betas'] = (0.5, 0.99)
    # ========================
    return hypers


part3_q1 = r"""
During generator training, we tweak the parameters in order to improve the classifier, and so it is 
useful to maintain the gradients during this section to track the improvement trajectory.
During discriminator training, in order to reach better performance for the classify operation,
we shouldn't change the generator's parameters and so we can just discard the gradients.
"""

part3_q2 = r"""
                                1.**No**
In this process the discriminator also improves (The generator and discriminator work together - the first generates
samples for the latter to classify). The early stopping will prevent further improvement
for the discriminator and possible further improvement for the generator, because if we cease the training based on the
generator's loss, the discriminator might classify the generator's output samples as fake images (the generator
could have improved more).

2.
When the generator's loss lowers, it generates realistic images with increasing success.

When the discriminator's loss is unchanged, it has similar success when classifying real images and generated images.

The situation presented in the question leads to the scenario where in each iteration, the generator improves
(hence generates better more realistic images), while the discriminator classifies the real and the generator's output
samples with similar success (similar loss) - hence also improving! (otherwise, we would have seen a diminish in its
results since the generator improves with his sample generated output - leading to harder to classify correctly samples
for the discriminator).

The bottom line is, when the generator improves and produces better harder to classify images and the discrimintaor loss
stays the same, it also improves - both the generator and discriminator are improving in this scenario.
"""

part3_q3 = r"""
VAE attempts to fit the samples in a gaussian distribution. 
This makes the images that are generated to be smooth and blurry.

The images generated by GAN are less smooth (noisy) but also sharp, because it learns how the sample data is 
distributed and tries to generate the results with a similar distribution.
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
