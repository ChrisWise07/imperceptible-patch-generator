import numpy as np

from math import floor, sqrt
from torch._C import device
from custom_dpatch_robust import RobustDPatch
from image_for_patch import Image_For_Patch
from constants import DEVICE, FRCNN
from data_plotter import Data_Plotter

from utils import (
    get_previous_steps,
    get_previous_training_data,
    record_attack_training_data,
    save_image_from_np_array,
    plot_predictions,
)

from main import (
    training_data_directory,
    final_patches_directory,
    final_patched_images_directory,
    final_predictions_images_directory,
    args,
)


def generate_adversarial_patch(
    attack: RobustDPatch, image: Image_For_Patch
) -> None:
    image_name = image.name
    data_plotter = Data_Plotter()
    image_copies = np.vstack([image.image_as_np_array])

    if args.previous_experiment_directory_name:
        previous_num_steps = get_previous_steps(
            args.previous_experiment_directory_name, image_name
        )
    else:
        previous_num_steps = 0

    image.append_to_training_progress_file(
        f"\n\n--- Generating adversarial patch for {image_name} ---"
    )

    for step in range(previous_num_steps, previous_num_steps + args.step_num):
        image.append_to_training_progress_file(
            f"\n\n--- Step Number: {step} ---"
        )

        # train adv patch to trick object detector and not to be perceptibile
        attack.generate(
            x=image_copies, print_nth_num=args.loss_print_freq, y=None
        )

        # Save adv patch and training data every step
        record_attack_training_data(attack, step + 1)

        data_plotter.save_training_data(
            attack.get_loss_tracker(),
            attack.get_perceptibility_learning_rate(),
            attack.get_detection_learning_rate(),
        )

        if (step + 1) % args.decay_freq == 0:
            attack.decay_detection_learning_rate()
            attack.decay_perceptibility_learning_rate()

    data_plotter.plot_training_data(image_name)


def generate_rrap_for_image(image_name, file_type):
    image = Image_For_Patch(
        name=image_name, object_detector=FRCNN, file_type=file_type
    )

    if args.previous_experiment_directory_name:
        previous_training_data = get_previous_training_data(
            args.previous_experiment_directory_name, image_name
        )
    else:
        previous_training_data = None

    training_data_path = (
        f"{training_data_directory}/training_data_for_{image_name}.txt"
    )

    image_height, image_width = (
        image.image_as_np_array.shape[1],
        image.image_as_np_array.shape[2],
    )

    crop_height = floor(image_height * (1 - sqrt(2 / 3)) * 0.5)
    crop_width = floor(image_width * (1 - sqrt(2 / 3)) * 0.5)

    attack = RobustDPatch(
        estimator=FRCNN,
        max_iter=args.max_iter,
        batch_size=1,
        verbose=False,
        rotation_weights=(1, 1, 0, 1),
        brightness_range=(0.4, 1.6),
        image_to_patch=image,
        training_data_path=training_data_path,
        dec_decay_rate=args.dec_decay_rate,
        percep_decay_rate=args.percep_decay_rate,
        detection_momentum=args.decep_mom,
        perceptibility_momentum=args.percep_mom,
        perceptibility_learning_rate=args.percep_lr,
        crop_range=(crop_height, crop_width),
        detection_learning_rate=args.decep_lr,
        previous_training_data=previous_training_data,
    )

    generate_adversarial_patch(attack, image)

    adv_patch = attack.get_patch()

    save_image_from_np_array(
        f"{final_patches_directory}/patch_for_{image_name}.{file_type}",
        adv_patch,
    )

    image_adv_as_np_array = attack.apply_patch(x=image.image_as_np_array)

    save_image_from_np_array(
        f"{final_patched_images_directory}/adv_{image_name}.{file_type}",
        image_adv_as_np_array[0],
    )

    predictions_class, predictions_boxes, predictions_score = plot_predictions(
        object_detector=FRCNN,
        image=image_adv_as_np_array,
        threshold=0.5,
        path=f"{final_predictions_images_directory}/adv_{image_name}.{file_type}",
    )

    image.append_to_training_progress_file(
        f"\n\n--- Final predictions for {image_name} with adversarial patch ---"
        + f"\npredicted classes: {str(predictions_class)}"
        + f"\npredicted score: {str(predictions_score)}"
    )

    return np.array(attack.get_loss_tracker().return_all_losses())
