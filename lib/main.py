import torch
from lib.renderer import Renderer
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from lib.core.module import CoreModule
from lib.metrics.loss_engine import LossEngine
from lib.models.rotation_encoder import RotationEncoder
from lib.models.encoder import resnet50
from lib.models.decoder import Decoder


class NeuralMaterial(CoreModule):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg

        self.rotation_encoder = RotationEncoder()
        self.encoder = resnet50(pretrained=True, num_classes=self.cfg.z)
        self.decoder = Decoder(self.cfg.w, self.cfg.z, self.cfg.layers)
        self.renderer = Renderer(self.cfg.renderer.fov, self.cfg.renderer.gamma, self.cfg.renderer.attenuation)
        self.loss = LossEngine(cfg.loss)

    def encode(self, image_in, mode):

        mu, logvar = self.encoder(image_in)

        if self.cfg.loss.kl > 0 and mode == 'train':
            std = logvar.mul(0.5).exp_()
            randn = torch.empty_like(std).normal_()
            z = mu + randn * std
        else:
            z = mu

        rot = self.rotation_encoder(image_in)

        return z, mu, logvar, rot

    def decode(self, z, x):

        decoding = self.decoder(z, x)

        brdf_maps = {
            'diffuse': decoding[:, :3],
            'specular': decoding[:, 3:4],
            'roughness': decoding[:, 4:5].clamp(0.01, 0.99),
            'normal': self.renderer.height_to_normal(decoding[:, 5:6])
        }

        return brdf_maps

    def forward(self, batch, mode, size=None):

        z, mu, logvar, rot = self.encode(batch, mode)

        if size is None:
            size = batch.shape[2:4]

        # sample noise
        x = torch.rand(z.size(0), self.cfg.w, *size, device=z.device)

        # convert noise to brdf maps using CNN
        brdf_maps = self.decode(z, x)

        # render brdf maps using differentiable rendering
        image_out = self.renderer(brdf_maps, rot_angle=rot, light_shift=None)

        return image_out, brdf_maps, z, mu, logvar

    def forward_step(self, batch, mode):
        # Extract the input image and ground-truth BRDF maps
        brdf_maps_gt = None

        image_in = batch[0]  # Input image
        if mode == 'test':
            image_in = image_in.unsqueeze(0)  # Adds a batch dimension at the 0th position

        if mode == 'train':
            brdf_maps_gt = batch[1]  # Ground-truth BRDF maps

        # Forward pass through the model
        image_out, brdf_maps, z, mu, logvar = self.forward(image_in, mode)

        # Compute losses (image and BRDF-specific)
        loss = self.loss(image_in, image_out, mu, logvar, self.global_step, brdf_maps, brdf_maps_gt)

        # Prepare outputs
        outputs = {
            'images': {'image_in': image_in, 'image_out': image_out, **brdf_maps},
            'metrics': loss
        }

        return outputs

    def configure_optimizer(self):

        return torch.optim.Adam(
            self.parameters(), self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

    def configure_optimizer_finetuning(self):

        return torch.optim.Adam(
            self.decoder.parameters(), self.cfg.lr * 10,
            weight_decay=self.cfg.weight_decay
        )