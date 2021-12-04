import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import cv2

from json import JSONEncoder
from torch import Tensor
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)
from typing import List, Tuple
from differential_color_functions import rgb2lab_diff, ciede2000_diff
from rrap_constants import *

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
                return obj.tolist()
        if isinstance(obj, Tensor):
                return obj.detach().numpy().tolist()
        return JSONEncoder.default(self, obj)

def save_image_from_np_array(path, np_array):
        plt.imsave(path, np.uint8(np_array))
        plt.close

def get_rgb_diff(image_tensor):
        return rgb2lab_diff(torch.stack([image_tensor], dim=0), DEVICE) 

def calculate_perceptibility_gradients_of_patched_image(og_image_rgb_diff, patched_image, loss_tracker):
        patch_image_tensor = TRANSFORM(patched_image).clamp(0,1).requires_grad_(True)
        patch_image_rgb_diff = get_rgb_diff(patch_image_tensor)
        d_map=ciede2000_diff(og_image_rgb_diff, patch_image_rgb_diff, DEVICE).unsqueeze(1)
        perceptibility_dis=torch.norm(d_map.view(1,-1),dim=1)
        perceptibility_loss = perceptibility_dis.sum()
        loss_tracker.update_perceptibility_loss(perceptibility_loss.item())
        perceptibility_loss.backward()
        return (patch_image_tensor.grad/torch.norm(patch_image_tensor.grad.view(1,-1),dim=1)).permute(1,2,0).numpy()

def get_perceptibility_gradients_of_patch(og_image_object, patched_image, loss_tracker):
        perceptibility_gradients = calculate_perceptibility_gradients_of_patched_image(og_image_object.image_rgb_diff, patched_image, loss_tracker)
        return perceptibility_gradients[
                og_image_object.patch_location[0]:og_image_object.patch_location[0] + og_image_object.patch_shape[0], 
                og_image_object.patch_location[1]:og_image_object.patch_location[1] + og_image_object.patch_shape[1], 
                ...
                ]

def file_handler(path, mode, func):
        try:
                with open(path, mode) as f:
                        value = func(f)
                return value
        except FileNotFoundError:
                return 0

def get_previous_training_data(previous_experiment_directory: str, image_name: str):
        training_data_path = f"{ROOT_EXPERIMENT_DATA_DIRECTORY}/{previous_experiment_directory}/training_data/training_data_for_{image_name}.txt"
        return file_handler(path = training_data_path, mode = "r", func = lambda f: json.load(f))

def get_previous_steps(previous_experiment_directory: str, image_name: str):
        training_data_path = f"{ROOT_EXPERIMENT_DATA_DIRECTORY}/{previous_experiment_directory}/training_data/training_data_for_{image_name}.txt"
        return file_handler(path = training_data_path, mode = "r", func = lambda f: json.load(f)["step_number"])

def record_attack_training_data(attack, step_number):
        training_data = {}
        training_data["step_number"] = step_number
        training_data["detection_learning_rate"] = attack.get_detection_learning_rate()
        training_data["perceptibility_learning_rate"] = attack.get_perceptibility_learning_rate()
        loss_tracker = attack.get_loss_tracker()
        training_data["loss_data"] = {"perceptibility_loss": loss_tracker.rolling_perceptibility_loss, 
                                      "detection_loss": loss_tracker.rolling_detection_loss}
        training_data["patch_np_array"] = attack.get_patch()
        training_data["old_patch_detection_update"] = np.array(attack.get_old_patch_detection_update())
        training_data["old_patch_perceptibility_update"] = np.array(attack.get_old_patch_perceptibility_update())
        file_handler(path = attack.get_training_data_path(), mode = "w", func = lambda f: json.dump(training_data, f, cls=NumpyArrayEncoder))


def plot_data(rolling_loss_history, current_loss_history, lr_history, image_name, loss_type):
        from rrap_main import loss_plots_directory
        # create figure and axis objects with subplots()
        fig,ax = plt.subplots()

        # set x-axis label
        ax.set_xlabel("Steps",fontsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        # make a plot
        ax.plot(rolling_loss_history, '-r', label=f'Rolling {loss_type} Loss')
        ax.plot(current_loss_history, '-g', label=f'Current {loss_type} Loss')

        # set y-axis label
        ax.set_ylabel(f"{loss_type} Loss", fontsize=10)
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # twin object for two different y-axis on the sample plot
        ax2=ax.twinx()
        # make a plot with different y-ax,is using second axis object
        ax2.plot(lr_history, '-b', label=f'{loss_type} Lr')

        # set y-axis label
        ax2.set_ylabel(f"{loss_type} Lr", fontsize=10)
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.legend(loc='upper left', prop={'size': 10})
        ax2.legend(loc='upper right', prop={'size': 10})

        # save the plot as a file
        plt.title(f"{loss_type} Data Over Step Numbers", fontsize=12)
        plt.savefig(f"{loss_plots_directory}/{loss_type}_loss_data_{image_name}", bbox_inches='tight')
        plt.close()

def plot_predictions(object_detector, image, path, threshold):     
        predictions_class, predictions_boxes, predictions_score = generate_predictions(object_detector, image, threshold)

        # Plot predictions
        plot_image_with_boxes(img=image[0].copy(), 
                              boxes=predictions_boxes, 
                              pred_cls=predictions_class, 
                              path = path)

        return predictions_class, predictions_boxes, predictions_score

def generate_predictions(object_detector, image: np.ndarray, threshold: float):
        #generate predictions
        predictions = object_detector.predict(x=image)

        # Process predictions   
        return extract_predictions(predictions[0], threshold)

def extract_predictions(predictions_, threshold):
        # Get the predicted class
        predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]

        # Get the predicted bounding boxes
        predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

        # Get the predicted prediction score
        predictions_score = list(predictions_["scores"])

        # Get a list of index with score greater than threshold
        predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

        predictions_boxes = predictions_boxes[: predictions_t + 1]
        predictions_class = predictions_class[: predictions_t + 1]
        predictions_score = predictions_score[: predictions_t + 1]

        return predictions_class, predictions_boxes, predictions_score
        
def plot_image_with_boxes(img: np.ndarray, boxes: List[tuple], pred_cls: List[str], path: str):
        text_size = 2
        text_th = 2
        rect_th = 2

        for i in range(len(boxes)):
                # Draw Rectangle with the coordinates
                cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])), 
                              color=(0, 255, 0), thickness=rect_th)

                # Write the prediction class
                cv2.putText(img, pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

        save_image_from_np_array(path, img)

def open_image_as_rgb_np_array(path: str) -> np.ndarray:
        return np.stack([cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)], axis=0).astype(np.float32)