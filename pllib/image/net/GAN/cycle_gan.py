import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.spectral_norm import spectral_norm
import torchaudio.transforms
import pytorch_lightning as pl

import matplotlib.pyplot as plt

from pllib.sound.utils import SoundSpecTransmitter, SoundIO


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=True):
        super(UpSample, self).__init__()
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=nn.InstanceNorm2d),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout_layer = nn.Dropout2d(0.5)

    def forward(self, x, shortcut=None):
        x = self.block(x)
        if self.dropout:
            x = self.dropout_layer(x)
        if shortcut is not None:
            x = torch.cat([x, shortcut], dim=1)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 apply_instancenorm=True, apply_spectral_norm=False):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=nn.InstanceNorm2d)
        if apply_spectral_norm:
            self.conv = spectral_norm(self.conv)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.apply_norm = apply_instancenorm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)
        return x


class UnetGenerator(nn.Module):
    def __init__(self, filter=64):
        super(UnetGenerator, self).__init__()
        self.downsamples = nn.ModuleList([
            DownSample(1, filter, kernel_size=4, apply_instancenorm=False),  # (b, filter, 128, 128)
            DownSample(filter, filter * 2),  # (b, filter * 2, 64, 64)
            DownSample(filter * 2, filter * 4),  # (b, filter * 4, 32, 32)
            DownSample(filter * 4, filter * 8),  # (b, filter * 8, 16, 16)
            DownSample(filter * 8, filter * 8),  # (b, filter * 8, 8, 8)
        ])

        self.upsamples = nn.ModuleList([
            UpSample(filter * 8, filter * 8),
            UpSample(filter * 16, filter * 4, dropout=False),
            UpSample(filter * 8, filter * 2, dropout=False),
            UpSample(filter * 4, filter, dropout=False)
        ])

        self.last = nn.Sequential(
            nn.ConvTranspose2d(filter * 2, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for l in self.downsamples:
            x = l(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for l, s in zip(self.upsamples, skips):
            x = l(x, s)
        out = self.last(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, filter=64):
        super(Discriminator, self).__init__()

        self.block = nn.Sequential(
            DownSample(1, filter, kernel_size=4, stride=2, apply_instancenorm=False, apply_spectral_norm=True),
            DownSample(filter, filter * 2, kernel_size=4, stride=2, apply_spectral_norm=True),
            DownSample(filter * 2, filter * 4, kernel_size=4, stride=2, apply_spectral_norm=True),
            DownSample(filter * 4, filter * 8, kernel_size=4, stride=1, apply_spectral_norm=True),
        )

        self.last = spectral_norm(nn.Conv2d(filter * 8, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x


class CycleGANModule(pl.LightningModule):
    def __init__(self, lr, transmitter: SoundSpecTransmitter, sound_io: SoundIO, reconstr_w=10, id_w=2):
        super(CycleGANModule, self).__init__()
        self.b2s_gen = UnetGenerator()
        self.s2b_gen = UnetGenerator()
        self.base_disc = Discriminator()
        self.style_disc = Discriminator()
        self.lr = lr
        self.reconstr_w = reconstr_w
        self.id_w = id_w
        self.transmitter = transmitter
        self.sound_io = sound_io

        self.mae = nn.L1Loss()
        self.generator_loss = nn.MSELoss()
        self.discriminator_loss = nn.MSELoss()

    def configure_optimizers(self):
        b2s_gen_optimizer = optim.Adam(self.b2s_gen.parameters(), lr=self.lr['G'])
        s2b_gen_optimizer = optim.Adam(self.s2b_gen.parameters(), lr=self.lr['G'])
        base_disc_optimizer = optim.Adam(self.base_disc.parameters(), lr=self.lr['D'])
        style_disc_optimizer = optim.Adam(self.style_disc.parameters(), lr=self.lr['D'])

        return [b2s_gen_optimizer, s2b_gen_optimizer,
                base_disc_optimizer, style_disc_optimizer], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        base_img, style_img = batch
        b, _, h, w = base_img.size()

        valid = torch.ones(b, 1, h // 8 - 2, w // 8 - 2).cuda()
        fake = torch.zeros(b, 1, h // 8 - 2, w // 8 - 2).cuda()

        # Train Generator
        if optimizer_idx == 0 or optimizer_idx == 1:
            # Validity
            # MSELoss
            val_base = self.generator_loss(self.base_disc(self.s2b_gen(style_img)), valid)
            val_style = self.generator_loss(self.style_disc(self.b2s_gen(base_img)), valid)
            val_loss = (val_base + val_style) / 2

            # Reconstruction
            reconstr_base = self.mae(self.s2b_gen(self.b2s_gen(base_img)), base_img)
            reconstr_style = self.mae(self.b2s_gen(self.s2b_gen(style_img)), style_img)
            reconstr_loss = (reconstr_base + reconstr_style) / 2

            # Identity
            id_base = self.mae(self.s2b_gen(base_img), base_img)
            id_style = self.mae(self.b2s_gen(style_img), style_img)
            id_loss = (id_base + id_style) / 2 * 0

            # Loss Weight
            gen_loss = self.id_w * val_loss + self.reconstr_w * reconstr_loss + self.id_w * id_loss
            self.log('g_loss', gen_loss, prog_bar=True, logger=True, on_step=True)
            self.log('validity', val_loss, prog_bar=True, logger=True, on_step=True)
            self.log('reconstr', reconstr_loss, prog_bar=True, logger=True, on_step=True)
            self.log('identity', id_loss, prog_bar=True, logger=True, on_step=True)
            return {'loss': gen_loss, 'validity': val_loss, 'reconstr': reconstr_loss, 'identity': id_loss}

        # Train Discriminator
        elif optimizer_idx == 2 or optimizer_idx == 3:
            # MSELoss
            base_gen_loss = self.discriminator_loss(self.base_disc(self.s2b_gen(style_img)), fake)
            style_gen_loss = self.discriminator_loss(self.style_disc(self.b2s_gen(base_img)), fake)
            base_valid_loss = self.discriminator_loss(self.base_disc(base_img), valid)
            style_valid_loss = self.discriminator_loss(self.style_disc(style_img), valid)
            gen_loss = (base_gen_loss + style_gen_loss) / 2

            # Loss Weight
            disc_loss = (gen_loss + base_valid_loss + style_valid_loss) / 3
            self.log('d_loss', disc_loss, prog_bar=True, logger=True, on_step=True)
            return {'loss': disc_loss}

    def training_epoch_end(self, outputs):
        # TODO: Values to be logged
        avg_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 4 for i in range(4)])
        G_mean_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        D_mean_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2 for i in [2, 3]])
        validity = sum([torch.stack([x['validity'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        reconstr = sum([torch.stack([x['reconstr'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        identity = sum([torch.stack([x['identity'] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        self.log('avg_loss', avg_loss, logger=True, prog_bar=True, on_epoch=True)
        self.log('G_loss', G_mean_loss, logger=True, prog_bar=True, on_epoch=True)
        self.log('D_loss', D_mean_loss, logger=True, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        base, style = batch
        styled_img = self.b2s_gen(base)
        based_img = self.s2b_gen(style)
        rev_based_img = self.s2b_gen(styled_img)
        rev_styled_img = self.b2s_gen(based_img)
        return {'styled': styled_img, 'based': based_img, 'rev_styled': rev_styled_img, 'rev_based': rev_based_img}

    def test_epoch_end(self, outputs) -> None:
        base_out = torch.zeros((256, 1), device=self.device)
        style_out = torch.zeros((256, 1), device=self.device)
        rev_base_out = torch.zeros((256, 1), device=self.device)
        rev_style_out = torch.zeros((256, 1), device=self.device)
        for output in outputs:
            based, styled, rev_based, rev_styled = output['based'], output['styled'], output['rev_based'], output['rev_styled']
            based = torch.reshape(based, shape=(based.size()[2], -1))
            base_out = torch.cat([base_out, based], dim=1)
            styled = torch.reshape(styled, shape=(styled.size()[2], -1))
            style_out = torch.cat([style_out, styled], dim=1)
            rev_based = torch.reshape(rev_based, shape=(rev_based.size()[2], -1))
            rev_base_out = torch.cat([rev_base_out, rev_based], dim=1)
            rev_styled = torch.reshape(rev_styled, shape=(rev_styled.size()[2], -1))
            rev_style_out = torch.cat([rev_style_out, rev_styled], dim=1)
        base_out = base_out.cpu().numpy()
        style_out = style_out.cpu().numpy()
        rev_base_out = rev_base_out.cpu().numpy()
        rev_style_out = rev_style_out.cpu().numpy()
        self.sound_io.save('base.wav', self.transmitter.decode(base_out))
        self.sound_io.save('style.wav', self.transmitter.decode(style_out))
        self.sound_io.save('rev_base.wav', self.transmitter.decode(rev_base_out))
        self.sound_io.save('rev_style.wav', self.transmitter.decode(rev_style_out))
        plt.imshow(base_out[:, 10000:11000])
        plt.show()
        plt.imshow(style_out[:, 10000:11000])
        plt.show()
        plt.imshow(rev_base_out[:, 10000:11000])
        plt.show()
        plt.imshow(rev_style_out[:, 10000:11000])
        plt.show()
