import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
import glob
import os
import cv2

from utils import match_image_range

class LocalizationDataset(Dataset):

    def __init__(self, dataset_path, shape=(64, 64), prior_dataset_path=None):
        super(LocalizationDataset, self).__init__()

        self.shape = shape


        data = sio.loadmat(dataset_path)
        noisy_frames = data['IQs']  # Shape: (128, 128, 800)
        clean_frames = data['GTs']  # Shape: (128, 128, 800)
        noise_mat_frames = data['NoiseMats']

        self.noisy_frames = np.transpose(noisy_frames, (2, 0, 1))
        self.clean_frames = np.transpose(clean_frames, (2, 0, 1))
        self.noise_mat_frames = np.transpose(noise_mat_frames, (2, 0, 1))

        if prior_dataset_path is not None:
            deep_loco_sim_images = np.load(prior_dataset_path)
            self.noisy_frames = match_image_range(self.noisy_frames, deep_loco_sim_images)
            self.clean_frames = match_image_range(self.clean_frames, deep_loco_sim_images)

    def __len__(self):
        return len(self.noisy_frames)

    def __getitem__(self, idx):
        noisy_resized = cv2.resize(self.noisy_frames[idx], self.shape, interpolation=cv2.INTER_AREA)
        clean_resized = cv2.resize(self.clean_frames[idx], self.shape, interpolation=cv2.INTER_AREA)
        noise_resized = cv2.resize(self.noise_mat_frames[idx], self.shape, interpolation=cv2.INTER_AREA)

        return (
            torch.tensor(noisy_resized, dtype=torch.float32),  # Add channel dim (1, 64, 64)
            torch.tensor(clean_resized, dtype=torch.float32),  # Add channel dim (1, 64, 64)
            torch.tensor(noise_resized, dtype=torch.float32)
        )



class ULM_Dataset(Dataset):
    """ PyTorch dataset for ULM denoising and feature extraction."""
    def __init__(self, dataset_path, shape=(64, 64), normalize=True):
        super(ULM_Dataset, self).__init__()
        self.dataset_path = dataset_path
        self.shape = shape

        # Load all .mat files
        # self.mat_files = glob.glob(os.path.join(dataset_path, "*.mat"))
        self.mat_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".mat")]
        # print(files)

        # Store references to all frames
        self.all_noisy = []
        self.all_clean = []
        self.all_noise_mats = []

        for mat_file in self.mat_files:
            # print(mat_file)
            try:
                data = sio.loadmat(mat_file)  # Load .mat file
                noisy_frames = data['IQs']  # Shape: (128, 128, 800)
                clean_frames = data['GTs']  # Shape: (128, 128, 800)
                noise_mat_frames = data['NoiseMats']
            except:
                continue

            # Reshape from (128, 128, 800) → (800, 128, 128)
            noisy_frames = np.transpose(noisy_frames, (2, 0, 1))
            clean_frames = np.transpose(clean_frames, (2, 0, 1))
            noise_mat_frames = np.transpose(noise_mat_frames, (2, 0, 1))

            self.all_noisy.append(noisy_frames)
            self.all_clean.append(clean_frames)
            self.all_noise_mats.append(noise_mat_frames)

        # Convert lists to NumPy arrays
        # print(self.all_noisy)
        self.all_noisy = np.concatenate(self.all_noisy, axis=0)  # Shape (N, 128, 128)
        # print(self.all_noisy.shape)
        self.all_clean = np.concatenate(self.all_clean, axis=0)  # Shape (N, 128, 128)
        self.all_noise_mats = np.concatenate(self.all_noise_mats, axis=0)  # Shape (N, 128, 128)

        if normalize:
            self.all_noisy = (self.all_noisy - np.min(self.all_noisy)) / (np.max(self.all_noisy) - np.min(self.all_noisy))
            self.all_clean = (self.all_clean - np.min(self.all_clean)) / (np.max(self.all_clean) - np.min(self.all_clean))

        # Convert to PyTorch tensors
        # self.all_noisy = torch.tensor(self.all_noisy, dtype=torch.float32)  # Add channel dim
        # self.all_clean = torch.tensor(self.all_clean, dtype=torch.float32)
        # self.all_noise_mats = torch.tensor(self.all_noise_mats, dtype=torch.float32)

    def __len__(self):
        return len(self.all_noisy)

    def __getitem__(self, idx):
        noisy_resized = cv2.resize(self.all_noisy[idx], self.shape, interpolation=cv2.INTER_AREA)
        clean_resized = cv2.resize(self.all_clean[idx], self.shape, interpolation=cv2.INTER_AREA)
        noise_resized = cv2.resize(self.all_noise_mats[idx], self.shape, interpolation=cv2.INTER_AREA)

        return (
            torch.tensor(noisy_resized, dtype=torch.float32),  # Add channel dim (1, 64, 64)
            torch.tensor(clean_resized, dtype=torch.float32),  # Add channel dim (1, 64, 64)
            torch.tensor(noise_resized, dtype=torch.float32)
        )

        # return (
        #     torch.tensor(self.all_noisy[idx], dtype=torch.float32).unsqueeze(0),  # Add channel dim (1, 64, 64)
        #     torch.tensor(self.all_clean[idx], dtype=torch.float32).unsqueeze(0),  # Add channel dim (1, 64, 64)
        #     torch.tensor(self.all_noise_mats[idx], dtype=torch.float32).unsqueeze(0)
        # )

# Example Usage
if __name__ == "__main__":
    dataset_path = "data/Data_Chick_Embryo"  # Update with your dataset path

    dataset = ULM_Dataset(dataset_path)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Check dataset size
    print(f"Dataset size: {len(dataset)}")

    # Retrieve a sample
    sample_noisy, sample_clean, sample_noise = dataset[0]
    print(f"Sample shape: {sample_noisy.shape}, {sample_clean.shape}, {sample_noise.shape}")
