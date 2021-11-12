import sys
import os 

from dataclasses import dataclass, field
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from rrap_utils import plot_data

@dataclass(frozen = True, repr=False, eq=False)
class Data_Plotter:
    rolling_perceptibility_loss_history: List[float] = field(default_factory=list)
    current_perceptibility_loss_history: List[float] = field(default_factory=list)
    perceptibility_lr_history: List[int] = field(default_factory=list)

    rolling_detection_loss_history: List[float] = field(default_factory=list)
    current_detection_loss_history: List[float] = field(default_factory=list)
    detection_lr_history: List[int] = field(default_factory=list)

    def save_training_data(self, loss_tracker, perceptibility_lr, detection_lr):
        self.rolling_perceptibility_loss_history.append(loss_tracker.rolling_perceptibility_loss)
        self.current_perceptibility_loss_history.append(loss_tracker.current_perceptibility_loss)
        self.perceptibility_lr_history.append(perceptibility_lr)

        self.rolling_detection_loss_history.append(loss_tracker.rolling_detection_loss)
        self.current_detection_loss_history.append(loss_tracker.current_detection_loss)
        self.detection_lr_history.append(detection_lr)

    def plot_training_data(self, image_name):
        plot_data(self.rolling_detection_loss_history, self.current_detection_loss_history, 
                  self.detection_lr_history, image_name, 'Detection')
        plot_data(self.rolling_perceptibility_loss_history, self.current_perceptibility_loss_history, 
                  self.perceptibility_lr_history, image_name, 'Perceptibility')