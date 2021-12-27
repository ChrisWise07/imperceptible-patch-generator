import argparse
import os
import json
import numpy as np

from constants import ROOT_EXPERIMENT_DATA_DIRECTORY, ROOT_DIRECTORY, IMAGES_DIRECTORY
from utils import file_handler
from performance_eval import mAP_calculator

parser = argparse.ArgumentParser(description='Process hyperparameters')
parser.add_argument('--max_iter', type=int, default=1000, help='Number of iterations per steps (default = 1000)') 
parser.add_argument('--step_num', type=int, default=50, help='Number of steps to perform (default = 50)')
parser.add_argument('--decep_lr', type=float, default=0.1, help='The deception learning rate (default = 0.1)')
parser.add_argument('--percep_lr', type=float, default=0.5, help='The perception learning rate (default = 0.5)')
parser.add_argument('--decay_rate', type=float, default=0.95, help='The amount learning rates are decayed(default = 0.95)')
parser.add_argument('--decay_freq', type=int, default=5, help='How frequent learnings rates are decayed (default = 5 steps)')
parser.add_argument('--decep_mom', type=float, default=0.9, help='The deception update momentum (default = 0.9)')
parser.add_argument('--percep_mom', type=float, default=0.9, help='The perception update momentum (default = 0.9)')
parser.add_argument('--data_folder_name', type=str, default=None, help='Name of folder which data sits inside (default = None)')
parser.add_argument('--previous_experiment_directory_name', type=str, default=None, help='Name of previous experiment directory; this is used to access previous training data (default = None)')
parser.add_argument('--patch_config', type=str, default='is', help='Starting patch configuration [random (r), image segment (is), black (b)] (default = is)')
parser.add_argument('--loss_print_freq', type=int, default=1000, help='Frequency, based on iteration number, to print losses (default = 1000)')
parser.add_argument('--dec_update_freq', type=int, default=1, help='Frequency, based on iteration number, to perform detection updates (default = 1)')

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

file_handler(path = f"{current_experiment_data_directory}/hyperparameters.txt", mode = "w", 
             func = lambda f: f.write(json.dumps(vars(args), indent=4)))

ground_truths = file_handler(path=f"{ROOT_DIRECTORY}/ground_truths.txt", mode="r", 
                             func=lambda f: json.load(f))

num_of_examples = len(ground_truths)

confidence_thresholds = [0.001, 0.1, 0.5]
mAP_calculators = [mAP_calculator(confidence_threshold=threshold, number_of_images=num_of_examples) for threshold in confidence_thresholds]

def main():
    from patch_generator import generate_rrap_for_image

    loss_names = ["Average detection loss", 
                 "Average rolling detection loss", 
                 "Average PerC distance", 
                 "Average rolling PerC distance"]
    loss_totals = np.zeros(4)

    file_name_type = [name.split(".") for name in os.listdir(IMAGES_DIRECTORY)]

    for name, type in file_name_type:
        loss_totals = np.add(loss_totals, generate_rrap_for_image(name, type))

        adv_image_path = f"{final_patched_images_directory}/adv_{name}.{type}"
        [calculator.map_confidence_to_tp_fp(ground_truths[name], adv_image_path) for calculator in mAP_calculators]

    results_string = "--- Average detection losses & PerC distances ---"
    for loss_name, loss_total in zip(loss_names, loss_totals):
        results_string += f"\n{loss_name}: {loss_total/num_of_examples}"       

    results_string += "\n\n--- mAP Results ---"
    total_mAP = 0.0
    for calculator in mAP_calculators:
        results_string += f"\nThe mAP for confidence threshold {calculator.confidence_threshold} = {calculator.calculate_mAP()}"
        total_mAP += calculator.mAP
    results_string += f"\nThe mAP averaged across all confidence thresholds = {total_mAP/len(mAP_calculators)}"

    file_handler(path=f"{current_experiment_data_directory}/experiment_results.txt", mode="w", 
                 func=lambda f: f.write(results_string))

if __name__ == "__main__":
    from timeit import default_timer as timer
    start = timer()
    main()
    print(timer() - start)