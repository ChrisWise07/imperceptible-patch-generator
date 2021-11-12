import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import json
import os

from json import JSONEncoder
from torch import Tensor
from PIL import Image
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from differential_color_functions import rgb2lab_diff, ciede2000_diff
from rrap_constants import *

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
                return obj.tolist()
        if isinstance(obj, Tensor):
                return obj.detach().numpy().tolist()
        return JSONEncoder.default(self, obj)

def save_image_from_np_array(np_array, path):
        Image.fromarray(np.uint8(np_array)).save(path)

def get_rgb_diff(image_tensor):
        return rgb2lab_diff(torch.stack([image_tensor], dim=0), DEVICE) 

def calculate_perceptibility_gradients_of_patch(og_image_patch_section_rgb_diff, patch, loss_tracker):
        patch_tensor = TRANSFORM(patch).clamp(0,1).requires_grad_(True)
        patch_rgb_diff = get_rgb_diff(patch_tensor)
        d_map=ciede2000_diff(og_image_patch_section_rgb_diff, patch_rgb_diff, DEVICE).unsqueeze(1)
        perceptibility_dis=torch.norm(d_map.view(1,-1),dim=1)
        perceptibility_loss = perceptibility_dis.sum()
        loss_tracker.update_perceptibility_loss(perceptibility_loss.item())
        perceptibility_loss.backward()
        return patch_tensor.grad.permute(1,2,0).numpy()
        
def file_handler(path, mode, func):
        try:
                with open(path, mode) as f:
                        value = func(f)
                return value
        except FileNotFoundError:
                return 0

def get_previous_training_data(training_data_path):
        return file_handler(path = training_data_path, mode = "r", func = lambda f: json.load(f))

def get_previous_steps(training_data_path):
        return file_handler(path = training_data_path, mode = "r", func = lambda f: int(json.load(f)["step_number"]))

def record_attack_training_data(attack, step_number):
        training_data = {}
        training_data["step_number"] = str(step_number)
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
        plt.savefig(f"{PLOTS_DIRECTORY}{loss_type}_loss_data_{image_name}", bbox_inches='tight')
        plt.close()