import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        modules = []

        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======

        modules.append(
            nn.Conv2d(in_size[0], 128, stride=2, padding=1,
                      kernel_size=4, bias=False))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.LeakyReLU(0.3))

        modules.append(nn.Conv2d(128, 256, stride=2, padding=1,
                                 kernel_size=4, bias=False))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.LeakyReLU(0.3))

        modules.append(nn.Conv2d(256, 512, stride=2, padding=1,
                                 kernel_size=4, bias=False))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.LeakyReLU(0.3))

        modules.append(nn.Conv2d(512, 1024, stride=2, padding=1,
                                 kernel_size=4, bias=False))
        modules.append(nn.BatchNorm2d(1024))
        modules.append(nn.LeakyReLU(0.3))

        # ========================
        self.disc = nn.Sequential(*modules)
        self.reg = nn.Linear(in_size[1] * in_size[2] * 4, 1)

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        f = self.disc(x)
        # ========================
        return self.reg(f.view(x.shape[0], -1))


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim
        modules = []

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules.append(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=1024,
                               kernel_size=featuremap_size, stride=1,
                               padding=0, bias=False))
        modules.append(nn.BatchNorm2d(1024))
        modules.append(nn.ReLU())
        modules.append(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, padding=1,
                               stride=2, bias=False))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.ReLU())
        modules.append(
            nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1,
                               stride=2, bias=False))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.ReLU())
        modules.append(
            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1,
                               stride=2, bias=False))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())
        modules.append(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, padding=1,
                               stride=2, bias=False))
        self.gen = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        with torch.set_grad_enabled(with_grad):
            samples = self.forward(torch.randn((n, self.z_dim), device=device))
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        # ========================
        return self.gen(z.view(z.shape[0], -1, 1, 1))


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======

    real_n = torch.ones(y_data.shape).to(y_data.device).uniform_(
        data_label - label_noise / 2,
        data_label + label_noise / 2)
    gen_n = torch.ones(y_generated.shape).to(y_generated.device).uniform_(
        1 - data_label - label_noise / 2,
        1 - data_label + label_noise / 2)
    loss_data, loss_generated = nn.BCEWithLogitsLoss()(y_data,
                                                       real_n), \
                                nn.BCEWithLogitsLoss()(
                                    y_generated, gen_n)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    # ========================
    return nn.BCEWithLogitsLoss()(y_generated,
                                  torch.ones_like(y_generated) * data_label)


def train_batch(
        dsc_model: Discriminator,
        gen_model: Generator,
        dsc_loss_fn: Callable,
        gen_loss_fn: Callable,
        dsc_optimizer: Optimizer,
        gen_optimizer: Optimizer,
        x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    x_gen = gen_model.sample(x_data.shape[0], with_grad=True)
    dsc_loss = dsc_loss_fn(dsc_model(x_data), dsc_model(x_gen.detach()))
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    gen_loss = gen_loss_fn(
        dsc_model(gen_model.sample(x_data.shape[0], with_grad=True)))
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if len(dsc_losses) >= 3 and gen_losses[-2] > gen_losses[-1]:
        torch.save(gen_model, checkpoint_file)
        saved = True
    # ========================

    return saved
