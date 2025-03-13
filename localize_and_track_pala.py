import argparse

from ULM.Localization2D import ULM_localization2D
from ULM.Tracking2D import ULM_tracking2D_PALA
from utils import visualize_tracking_results, load_mat_file


def main():
    parser = argparse.ArgumentParser(description="Run ULM localization and tracking.")
    parser.add_argument("--test_data_path", type=str, default="data/Test_Localization/cam4_part1_patch13.mat",
                        help="Path to the test dataset.")
    parser.add_argument("--plot_save_path", type=str, default="tracking_results.png",
                        help="Path to save the visualization output.")

    args = parser.parse_args()

    noisy_frames, clean_frames, noise_mat_frames = load_mat_file(args.test_data_path)

    # ULM Parameters
    ULM_params = {
        'fwhm': (20, 20),
        'NLocalMax': 10,
        'LocMethod': 'wa',
        'InterpMethod': 'spline',
        'numberOfParticles': 30,
        'max_linking_distance': 30,
        'max_gap_closing': 5,
        'min_length': 5,
        'scale': (1, 1, 0.01),
        'res': 1
    }

    MatTracking = ULM_localization2D(noisy_frames, ULM_params)

    raw_tracks, interpolated_tracks = ULM_tracking2D_PALA(MatTracking, ULM_params)

    visualize_tracking_results(noisy_frames, MatTracking, raw_tracks, interpolated_tracks, args.plot_save_path)


if __name__ == "__main__":
    main()
