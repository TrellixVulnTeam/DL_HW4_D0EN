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
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TODO: Go over this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TODO: Go over this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TODO: Go over this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TODO: Go over this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                                    **1**

                                <loss_p>
'epg' and 'vpg' begin with a negative policy loss of around 
-90 which rises towards 0+ as the episodes progress, as these methods do not 
use the baseline reduction. The other methods, 'bpg' and 'cpg' do use 
baseline reduction which is why the graph shows subtle changes around 0 in 
the policy loss, which is a direct affect of reducing the baseline.


                                <baseline>
Only 'bpg' and 'cpg' use baseline reduction which is why 
the other methods are not in this graph. Both methods begin with a baseline 
of around -60 which rises towards $10$ as the episodes progress. Bigger 
baseline values assist the network in increasing the reward.


                                <loss_e>
Since only 'epg' and 'cpg' use entropy loss, the other models 
are not in this graph. Both the 'epg' and 'cpg' begin with a negative entropy 
loss of around -0.5 which rises towards zero as the episodes progress. A 
higher absolute value here indicates that the probability distribution is 
close to a uniform one, while a lower absolute value indicates convergence of 
the network to a better policy. In 'cpg' there is a baseline subtraction 
which causes a faster convergence than in 'epg', which is why 'cpg' is a 
better choice here. 


                                <mean_reward>
While all methods start with a reward of -200 and rise 
as the episodes progress, there is a difference in how fast each method's 
reward rises. As can be seen in the graph, the methods which use a baseline 
rise a lot faster and higher than those which don't use a baseline. 




                                    **2**

                                <loss_p>
Although starting with the same policy loss, 'aac' rises a 
lot more quickly than 'vpg' and 'epg', resulting in a lower trajectory loss. 
Therefore 'aac' is much better than both 'vpg' and 'epg' in terms of policy 
loss as and thus can get a better approximation of the state value.

                                <baseline>
'aac' does not use a baseline reduction so the baseline 
graph stays the same as before.

                                <loss_e> 
In terms of absolute value, 'aac' has both a lower entropy 
loss and lowers faster than the entropy loss of 'epg' and 'cpg', making it 
clearly the better choice here. 

                                <mean_reward>
Here too, 'aac' clearly outperforms all other methods as 
it reaches a mean_reward of around ~$130$. Although in the first ~$1000$ 
episodes 'aac' performed worse and even decreased, starting from the $1000$ 
episode it rose a lot faster than all other methods, reaching a higher 
mean_reward than all others in episode ~1200. 
"""

# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
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
    raise NotImplementedError()
    # ========================
