from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.io as io
import torch
import logging


class MaterialDataset(Dataset):

    def __init__(self, cfg, mode):
        """
        Args:
            cfg: Configuration object/dictionary containing dataset path and image size.
            mode: 'train' or 'test' to indicate which split to load.
        """
        # Initialize logger
        self.logger = logging.getLogger(f"MaterialDataset_{mode}")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        self.cfg = cfg
        self.mode = mode
        self.path_dir = Path(cfg.path) / mode  # train or test folder

        # Collect all samples based on folder structure
        self.samples = self.get_samples()

        if len(self.samples) == 0:
            self.logger.error(f"No samples found in {self.path_dir} for mode '{mode}'.")
            raise ValueError(f"No samples found in {self.path_dir} for mode '{mode}'.")

        self.dataset_length = len(self.samples)

        # Log dataset loading success
        # self.logger.info(f"Successfully loaded {self.dataset_length} samples from {self.path_dir}.")

        # Define transforms for the dataset
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(self.cfg.size[0]), int(self.cfg.size[1]))),
            transforms.RandomVerticalFlip(0.5) if mode == 'train' else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip(0.5) if mode == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.dataset_length

    def get_samples(self):
        """
        Traverse the dataset directory and collect all sample folders.
        Each folder should contain 'input.png' and a 'brdf' subfolder.
        """
        samples = []
        self.logger.info(f"Scanning dataset directory: {self.path_dir}")

        for sample_dir in self.path_dir.iterdir():
            if not sample_dir.is_dir():
                self.logger.warning(f"Skipping non-directory item: {sample_dir}")
                continue

            input_path = sample_dir / 'input.png'
            brdf_dir = sample_dir / 'brdf'

            if input_path.exists() and brdf_dir.exists():
                brdf_files = {
                    'normal': brdf_dir / 'normal.png',
                    'diffuse': brdf_dir / 'diffuse.png',
                    'roughness': brdf_dir / 'roughness.png',
                    'specular': brdf_dir / 'specular.png'
                }

                # Ensure all BRDF maps exist
                if all(f.exists() for f in brdf_files.values()):
                    samples.append({'input': input_path, 'brdf': brdf_files})
                    # self.logger.info(f"Loaded sample: {sample_dir}")
                else:
                    self.logger.warning(f"Missing BRDF maps in: {sample_dir}")
            else:
                self.logger.warning(f"Missing input or brdf folder in: {sample_dir}")

        return samples

    def __getitem__(self, idx):
        """
        Load and return the input image and BRDF maps for a given sample.
        """
        sample = self.samples[idx]

        # Load the input image
        input_image = io.read_image(str(sample['input']))
        input_image = self.transforms(input_image)

        # Load the BRDF maps
        brdf_maps = {}
        for map_name, map_path in sample['brdf'].items():
            map_image = io.read_image(str(map_path))
            brdf_maps[map_name] = self.transforms(map_image)

        # Optionally shuffle the RGB channels of the input image (if required)
        if self.mode == 'train':
            shuffle_idx = torch.randperm(3)
            input_image = input_image[shuffle_idx, ...]

        return input_image, brdf_maps
