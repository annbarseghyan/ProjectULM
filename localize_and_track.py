import argparse

import torch

from DeepLoco.networks import DeepLoco
from DeepLoco.Dataset import LocalizationDataset
import ULM.TrackingNew as T
from utils import *


def localize_2d(model, dataset, threshold, device):
    """
    DeepLoco Localization function with
    Parameters
    ----------
    model: DeepLoco
    dataset: LocalizationDataset
        contains one ULM image (n_frames, h, w)
    threshold: float
        threshold for filtering probabilities
    device: torch.device

    Returns
    -------
        torch.Tensor
        A tensor containing detected slices formatted for tracking, with structure:
        `[[[frame_0, x_0_0, y_0_0], [frame_0, x_0_1, y_0_1], ...], ...]`
    """

    all_noisy = torch.stack([dataset[i][0] for i in range(len(dataset))])

    with torch.no_grad():
        points = process_model_output(model, all_noisy.to(device), threshold)
    return points


def track_2d(formatted_points, max_linking_distance, max_gap_closing, plot_save_path):
    """
    Function for creating tracks and visualizing them in an image
    Parameters
    ----------
    formatted_points: list
        list returned by localize_2d function
    max_linking_distance: int
        The maximum distance allowed for linking points between frames.
    max_gap_closing: int
        The maximum number of frames allowed for gap closing in tracking.
    plot_save_path: str
        path for saving result image
    """

    tracks, index_tracks = T.simple_tracker(formatted_points, max_linking_distance, max_gap_closing)
    T.show_tracking_simple(tracks, plot_save_path)


def main():
    parser = argparse.ArgumentParser(description="Run localization and tracking using DeepLoco model.")

    parser.add_argument("--model_path", type=str, default="adapted_data_simulation.pth", help="Path to the trained model.")
    parser.add_argument("--test_data_path", type=str, default="data/Test_Localization/cam4_part1_patch13.mat", help="Path to the test dataset.")
    parser.add_argument("--sim_images_path", type=str, default="deeploco_simulated_images.npy", help="Path to simulated images for prior dataset.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Probability threshold for localization.")
    parser.add_argument("--max_linking_distance", type=float, default=10.0, help="Maximum distance for linking points in tracking.")
    parser.add_argument("--max_gap_closing", type=int, default=5, help="Maximum gap (in frames) that can be closed in tracking.")
    parser.add_argument("--plot_save_path", type=str, default="tracking_results.png", help="Path to save the tracking plot.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    print(f"Using device: {device}")

    test_dataset = LocalizationDataset(
        dataset_path=args.test_data_path,
        shape=(64, 64),
        prior_dataset_path=args.sim_images_path
    )

    model = DeepLoco(min_coords=[0, 0], max_coords=[6400, 6400]).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    formatted_points = localize_2d(model, test_dataset, args.threshold, device)
    track_2d(formatted_points, args.max_linking_distance, args.max_gap_closing, args.plot_save_path)


if __name__ == "__main__":
    main()
