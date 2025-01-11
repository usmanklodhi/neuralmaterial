from .losses import *
import torch
import kornia


class LossEngine(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.vgg = VGGFeatures()
        self.gram_loss = GramLoss()
        self.vggps_loss = VGGPSLoss()

    def get_crops(self, image_in, image_out):

        bs, _, h, w = image_in.size()
        resample_size_h = h
        resample_size_w = w

        rand = torch.rand((1,)).item()
        start = self.cfg.crop[0]
        end = self.cfg.crop[1]
        zoom_factor = rand * (end - start) + start

        res = h * zoom_factor
        downscale = res / resample_size_h
        sigma = 2 * downscale / 6.0

        if zoom_factor > 1:
            image_in = kornia.filters.gaussian_blur2d(
                image_in, (5, 5), (sigma, sigma))
            image_out = kornia.filters.gaussian_blur2d(
                image_out, (5, 5), (sigma, sigma))

        grid = kornia.create_meshgrid(
            resample_size_h, resample_size_w,
            normalized_coordinates=True,
            device=torch.device(image_in.device)
        ).expand(bs, resample_size_h, resample_size_w, 2)

        grid = grid + 1 + torch.rand((1,)).item() * 2

        grid = (grid * zoom_factor) % 4.0
        grid = torch.where(grid > 2, 4 - grid, grid)
        grid = grid - 1

        crops_in = torch.nn.functional.grid_sample(
            image_in, grid, mode='bilinear', align_corners=True)

        crops_out = torch.nn.functional.grid_sample(
            image_out, grid, mode='bilinear', align_corners=True)

        return crops_in, crops_out

    def compute_brdf_losses(self, brdf_maps, brdf_maps_gt):
        """
        Compute supervised losses between predicted and ground-truth BRDF maps.
        """
        losses = {}

        for map_name, pred_map in brdf_maps.items():
            gt_map = brdf_maps_gt[map_name]

            # Debugging: Log shapes
            # print(f"[DEBUG] Processing BRDF map: '{map_name}'")
            # print(f"[DEBUG] Predicted map shape: {pred_map.shape}, dtype: {pred_map.dtype}")
            # print(f"[DEBUG] Ground-truth map shape: {gt_map.shape}, dtype: {gt_map.dtype}")

            # Align shapes if necessary
            if pred_map.shape[1] < gt_map.shape[1]:  # Expand predicted map
                pred_map = pred_map.expand(-1, gt_map.shape[1], -1, -1)
                # print(f"[DEBUG] Expanded predicted map to: {pred_map.shape}")
            elif pred_map.shape[1] > gt_map.shape[1]:  # Reduce ground-truth map
                gt_map = gt_map.mean(dim=1, keepdim=True)
                # print(f"[DEBUG] Reduced ground-truth map to: {gt_map.shape}")

            # Compute L1 loss
            l1_loss = torch.nn.functional.l1_loss(pred_map, gt_map)
            losses[f'{map_name}_loss'] = l1_loss
            # print(f"[DEBUG] '{map_name}' L1 loss: {l1_loss.item()}")

        # Total BRDF loss
        total_brdf_loss = sum(losses.values())
        losses['total_brdf_loss'] = total_brdf_loss
        # print(f"[DEBUG] Total BRDF loss: {total_brdf_loss.item()}")

        return losses

    def forward(self, image_in, image_out, mu, logvar, step, brdf_maps=None, brdf_maps_gt=None):
        # KL Loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Slowly increase KL loss
        N = 10000
        if step < N:
            kl_loss = (step / N) * kl_loss

        # Compute image-related losses
        crops_in, crops_out = self.get_crops(image_in, image_out)
        crops_in_vgg = self.vgg(crops_in)
        crops_out_vgg = self.vgg(crops_out)

        gram_loss = self.gram_loss(crops_out_vgg, crops_in_vgg)
        vggps_loss = self.vggps_loss(crops_out_vgg, crops_in_vgg)

        # BRDF Map Losses
        brdf_losses = {}
        if brdf_maps is not None and brdf_maps_gt is not None:
            brdf_losses = self.compute_brdf_losses(brdf_maps, brdf_maps_gt)

        # Total loss
        loss = (
                gram_loss * self.cfg.gram
                + vggps_loss * self.cfg.vggps
                + self.cfg.kl * kl_loss
                + brdf_losses.get('total_brdf_loss', 0.0) * self.cfg.brdf_weight
        )

        losses = {
            'loss': loss.mean(),
            'gram': gram_loss,
            'vggps': vggps_loss,
            'kl': kl_loss,
            **brdf_losses  # Include individual BRDF losses in the output
        }

        return losses

