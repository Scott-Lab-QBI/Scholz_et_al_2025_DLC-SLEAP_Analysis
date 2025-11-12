"""
    This script, when run as "python run_demo.py -c "path\to\config\file" " will process a folder
    containing pose-estimation results. See the config.yaml file for more details on the required metadata
    and parameters. 
"""
import sys
import argparse
from pathlib import Path
import yaml
import pandas as pd

# Construct the path to the 'src' directory and add it to sys.path
src_path = Path(__file__).resolve().parent.parent / 'src'
print(src_path)
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

import zf_track_analysis

if __name__ == '__main__':

    # configure argument parser
    AP = argparse.ArgumentParser()
    AP.add_argument("-c",
                    "--config",
                    required=True,
                    help="Path to the YAML configuration file.")

    ARGS = vars(AP.parse_args())

    # Load configuration from YAML file
    config_path = Path(ARGS['config'])
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Configuration loaded from: {config_path}")

    # Unpack config values for use in the script
    tracking_path = Path(config['tracking_folder_path'])

    exp_metadata = {
        'experiment_specs': config['experiment_specs'],
        'camera_specs': config['camera_specs']
    }
    bodyparts_dict = config['bodyparts_dict']
    metrics_params = config['metrics_params']
    bout_detection_params = config['bout_detection_params']
    bout_metrics_params = config['bout_metrics_params']
    fps = exp_metadata['camera_specs']['frame_rate']
    result_type = config['result_type']
    result_pattern = config['result_pattern']


    print(f"Searching for .h5 files in: {tracking_path}")

    file_list = list(tracking_path.glob('*' + result_pattern + '.h5'))

    print(f"     Found {len(file_list)} tracking files to process")

    for file in file_list:
        if result_type == 'SLEAP':
            df = zf_track_analysis.sleap_to_dlc_format(file)
        if result_type == 'DLC':
            df = pd.read_hdf(file)

        scorer = df.columns.get_level_values(0)[0]

        all_metrics = zf_track_analysis.get_all_metrics(df,
                                                   exp_metadata=exp_metadata,
                                                   bodyparts_dict=bodyparts_dict,
                                                   **metrics_params)

        print("     Finished computing frame-by-frame kinematic metrics. Detecting bouts now...")
        bouts = zf_track_analysis.bout_detector(all_metrics, frame_rate=fps, **bout_detection_params)

        bouts_df = zf_track_analysis.compute_bout_metrics(bouts,
                                                     all_metrics,
                                                     FPS=fps,
                                                     **bout_metrics_params)

        print(f"     Detected {len(bouts_df['onset'])} bouts in {file.name}.")

        bouts_df = pd.DataFrame.from_dict(bouts_df)

        # --- Save the processed data ---
        # Construct the output file paths
        ethogram_path = file.with_name(f"{file.stem}_ethogram.h5")
        bouts_path = file.with_name(f"{file.stem}_bouts.h5")

        # Save the all_metrics DataFrame
        all_metrics.to_hdf(ethogram_path, key='df', mode='w')
        print(f"        Saved frame-by-frame metrics to: {ethogram_path.name}")

        # Save the bouts_df DataFrame
        bouts_df.to_hdf(bouts_path, key='df', mode='w')
        print(f"        Saved bout metrics to: {bouts_path.name}")
