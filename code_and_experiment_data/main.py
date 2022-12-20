import argparse
import os
import json
from typing import List
import numpy as np

from constants import (
    ROOT_EXPERIMENT_DATA_DIRECTORY,
    ROOT_DIRECTORY,
    IMAGES_DIRECTORY,
)
from utils import file_handler
from performance_eval import mAP_calculator


def set_up_parser_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Sets up the parser arguments for the main.py file

    Args:
        parser (argparse.ArgumentParser): The parser to set up
    """
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Number of iterations per steps (default = 1000)",
    )
    parser.add_argument(
        "--step_num",
        type=int,
        default=50,
        help="Number of steps to perform (default = 50)",
    )
    parser.add_argument(
        "--decep_lr",
        type=float,
        default=0.1,
        help="The deception learning rate (default = 0.1)",
    )
    parser.add_argument(
        "--percep_lr",
        type=float,
        default=0.5,
        help="The perception learning rate (default = 0.5)",
    )
    parser.add_argument(
        "--dec_decay_rate",
        type=float,
        default=0.95,
        help="The amount the deception learning rate is decayed (default = 0.95)",
    )
    parser.add_argument(
        "--percep_decay_rate",
        type=float,
        default=0.95,
        help="The amount the perceptability learning rate is decayed (default = 0.95)",
    )
    parser.add_argument(
        "--decay_freq",
        type=int,
        default=5,
        help="The number of steps before the learning rates are decayed (default = 5)",
    )
    parser.add_argument(
        "--decep_mom",
        type=float,
        default=0.9,
        help="The deception update momentum (default = 0.9)",
    )
    parser.add_argument(
        "--percep_mom",
        type=float,
        default=0.9,
        help="The perception update momentum (default = 0.9)",
    )
    parser.add_argument(
        "--data_folder_name",
        type=str,
        default=None,
        help="Name of folder which data sits inside (default = None)",
    )
    parser.add_argument(
        "--previous_experiment_directory_name",
        type=str,
        default=None,
        help="Name of previous experiment directory; this is used to access previous training data (default = None)",
    )
    parser.add_argument(
        "--patch_config",
        type=str,
        default="is",
        help="Starting patch configuration [random (r), image segment (is), black (b)] (default = is)",
    )
    parser.add_argument(
        "--loss_print_freq",
        type=int,
        default=1000,
        help="Frequency, based on iteration number, to print losses (default = 1000)",
    )
    parser.add_argument(
        "--dec_update_freq",
        type=int,
        default=1,
        help="Frequency, based on iteration number, to perform detection updates (default = 1)",
    )


def make_directories(directory_names: List[str]) -> None:
    """
    Makes the directories for the current experiment

    Args:
        args (argparse.Namespace): The arguments to use
    """
    for directory_name in directory_names:
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)


parser = argparse.ArgumentParser(
    description="Imperceptible Patch Hyperparameters"
)

if not hasattr(parser, "set_up_parser_arguments"):
    set_up_parser_arguments(parser)

args = parser.parse_args()

list_of_directories = []

current_experiment_data_directory = (
    f"{ROOT_EXPERIMENT_DATA_DIRECTORY}/{args.data_folder_name}"
)
list_of_directories.append(current_experiment_data_directory)

initial_predictions_images_directory = (
    f"{current_experiment_data_directory}/initial_predictions"
)
list_of_directories.append(initial_predictions_images_directory)

final_patches_directory = f"{current_experiment_data_directory}/patches_adv"
list_of_directories.append(final_patches_directory)

final_predictions_images_directory = (
    f"{current_experiment_data_directory}/final_predictions"
)
list_of_directories.append(final_predictions_images_directory)

final_patched_images_directory = (
    f"{current_experiment_data_directory}/final_patched_images"
)
list_of_directories.append(final_patched_images_directory)

training_data_directory = f"{current_experiment_data_directory}/training_data"
list_of_directories.append(training_data_directory)

training_loss_printouts_directory = (
    f"{current_experiment_data_directory}/training_loss_printouts"
)
list_of_directories.append(training_loss_printouts_directory)

loss_plots_directory = (
    f"{current_experiment_data_directory}/loss_plots_directory"
)
list_of_directories.append(loss_plots_directory)

ground_truths = file_handler(
    path=f"{ROOT_DIRECTORY}/ground_truths.txt",
    mode="r",
    func=lambda f: json.load(f),
)

file_name_type = [name.split(".") for name in os.listdir(IMAGES_DIRECTORY)]

mAP_calculators = [
    mAP_calculator(
        confidence_threshold=threshold, number_of_images=len(file_name_type)
    )
    for threshold in [0.001, 0.1, 0.5]
]

loss_names = [
    "Average detection loss",
    "Average rolling detection loss",
    "Average PerC distance",
    "Average rolling PerC distance",
]


def main():
    from patch_generator import generate_rrap_for_image

    print("Starting imperceptible patch experiments ... \n")
    make_directories(list_of_directories)

    
    file_handler(
        path=f"{current_experiment_data_directory}/hyperparameters.txt",
        mode="w",
        func=lambda f: f.write(json.dumps(vars(args), indent=4)),
    )

    loss_totals = np.zeros(4)

    for name, file_type in file_name_type:
        loss_totals = np.add(
            loss_totals, generate_rrap_for_image(name, file_type)
        )

    results_string = "--- Average detection losses & PerC distances ---"
    for loss_name, loss_total in zip(loss_names, loss_totals):
        results_string += f"\n{loss_name}: {loss_total/len(file_name_type)}"

    [
        calculator.calculate_mAP(ground_truths, file_name_type)
        for calculator in mAP_calculators
    ]

    results_string += "\n\n--- mAP Results ---"
    total_mAP = 0.0
    for calculator in mAP_calculators:
        results_string += f"\nThe mAP for confidence threshold {calculator.confidence_threshold} = {calculator.mAP}"
        total_mAP += calculator.mAP
    results_string += f"\nThe mAP averaged across all confidence thresholds = {total_mAP/len(mAP_calculators)}"

    file_handler(
        path=f"{current_experiment_data_directory}/experiment_results.txt",
        mode="w",
        func=lambda f: f.write(results_string),
    )


if __name__ == "__main__":
    main()
