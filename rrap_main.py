import argparse
import os

from rrap_constants import ROOT_EXPERIMENT_DATA_DIRECTORY, IMAGES_DIRECTORY

parser = argparse.ArgumentParser(description='Process hyperparameters')

parser.add_argument('--max_iter', type=int, default=1000, help='Number of iterations per steps (default = 1000)') 
parser.add_argument('--step_num', type=int, default=50, help='Number of steps to perform (default = 50)')
parser.add_argument('--decep_lr', type=float, default=0.1, help='The deception learning rate (default = 0.1)')
parser.add_argument('--percep_lr', type=float, default=0.5, help='The perception learning rate (default = 0.5)')
parser.add_argument('--decay_rate', type=float, default=0.95, help='How much the deception is decayed (default = 0.95)')
parser.add_argument('--decep_mom', type=float, default=0.9, help='The deception update momentum (default = 0.9)')
parser.add_argument('--percep_mom', type=float, default=0.9, help='The perception update momentum (default = 0.9)')
parser.add_argument('--data_folder_name', type=str, default=None, help='Name of folder which data sits inside (default = None)')
parser.add_argument('--previous_experiment_directory_name', type=str, default=None, help='Name of previous experiment directory; this is used to access previous training data (default = None)')

args = parser.parse_args()

def create_experiment_data_directory(path: str):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

current_experiment_data_directory = f"{ROOT_EXPERIMENT_DATA_DIRECTORY}/{args.data_folder_name}"
initial_predictions_images_directory = f"{current_experiment_data_directory}/initial_predictions"
final_patches_directory = f"{current_experiment_data_directory}/patches_adv"
final_predictions_images_directory = f"{current_experiment_data_directory}/final_predictions"
final_patched_images_directory = f"{current_experiment_data_directory}/final_patched_images"
training_data_directory = f"{current_experiment_data_directory}/training_data"
training_loss_printouts_directory = f"{current_experiment_data_directory}/training_loss_printouts"
loss_plots_directory = f"{current_experiment_data_directory}/loss_plots_directory"

create_experiment_data_directory(current_experiment_data_directory)
create_experiment_data_directory(initial_predictions_images_directory)
create_experiment_data_directory(final_patches_directory)
create_experiment_data_directory(final_predictions_images_directory)
create_experiment_data_directory(final_patched_images_directory)
create_experiment_data_directory(training_data_directory)
create_experiment_data_directory(training_loss_printouts_directory)
create_experiment_data_directory(loss_plots_directory)

def main():
    from rrap_patch_generator import generate_rrap_for_image

    with os.scandir(IMAGES_DIRECTORY) as entries:
        [generate_rrap_for_image(entry.name) for entry in entries]

if __name__ == "__main__":
    main()