import numpy as np
import cv2

from dataclasses import InitVar, dataclass, field
from torch.functional import Tensor
from typing import Tuple
from rrap_utils import get_rgb_diff, file_handler, save_image_from_np_array
from rrap_constants import *

@dataclass(repr=False, eq=False)
class Image_For_Patch:
    name: str
    file_type: InitVar[str] 
    object_detector: InitVar[None]
    image_as_np_array: np.ndarray = field(init=False)
    patch_size: Tuple[int, int, int] = field(init=False)
    patch_location: Tuple[int, int] = field(init=False)
    patch_section_of_image: np.ndarray = field(init=False)
    image_rgb_diff: Tensor = field(init=False)

    def __post_init__(self, file_type, object_detector):
        self.image_as_np_array = self.open_image_as_rgb_np_array(file_type)
        prediction_box = self.generate_predictions_for_image(object_detector, self.image_as_np_array, path = f"{INITIAL_PREDICTIONS_DIRECTORY}{self.name}.{file_type}")

        #Customise patch location to centre of prediction box and patch to ratio of prediction box
        self.patch_shape, self.patch_location = self.cal_custom_patch_shape_and_location(prediction_box)

        self.patch_section_of_image = self.image_as_np_array[0][self.patch_location[0]:self.patch_location[0] + self.patch_shape[0], 
                                                                self.patch_location[1]:self.patch_location[1] + self.patch_shape[1], 
                                                                ...]

        #Calculate RGB perceptability of image
        #top_left_x, top_left_y, top_right_x, top_right_y = prediction_box[0][0], prediction_box[0][1], prediction_box[1][0], prediction_box[1][1] 
        #prediction_box_section_of_image = self.image_as_np_array[0][int(top_left_y):int(top_right_y), int(top_left_x):int(top_right_x), :]
        #save_image_from_np_array(prediction_box_section_of_image, f"{DATA_DIRECTORY}{self.name}.{file_type}")
        self.image_rgb_diff = get_rgb_diff(TRANSFORM(self.image_as_np_array[0]).clamp(0,1))
    
    def open_image_as_rgb_np_array(self, file_type) -> np.ndarray:
        img = cv2.cvtColor(cv2.imread(f"{IMAGES_DIRECTORY}{self.name}.{file_type}"), cv2.COLOR_BGR2RGB)
        return np.stack([img], axis=0).astype(np.float32)

    def generate_predictions_for_image(self, object_detector, image, path):
        self.append_to_training_progress_file(f"\n--- Initial Predictions for {self.name} ---")
        predictions_boxes = self.plot_predictions(object_detector, image, path) 
                                            
        return predictions_boxes

    def plot_predictions(self, object_detector, image, path):     
        predictions_class, predictions_boxes = self.generate_predictions(object_detector, image)

        # Plot predictions
        self.plot_image_with_boxes(img=image[0].copy(), 
                              boxes=predictions_boxes, 
                              pred_cls=predictions_class, 
                              path = path)

        return predictions_boxes[0]

    def generate_predictions(self, object_detector, image):
        #generate predictions
        predictions = object_detector.predict(x=image)

        # Process predictions   
        return self.extract_predictions(predictions[0])

    def extract_predictions(self, predictions_,):
        # Get the predicted class
        predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]

        # Get the predicted bounding boxes
        predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

        # Get the predicted prediction score
        predictions_score = list(predictions_["scores"])
        
        self.append_to_training_progress_file(f"predicted classes: {str(predictions_class)} \npredicted score: {str(predictions_score)}")

        # Get a list of index with score greater than threshold
        threshold = 0.5
        predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

        predictions_boxes = predictions_boxes[: predictions_t + 1]
        predictions_class = predictions_class[: predictions_t + 1]

        return predictions_class, predictions_boxes
    
    def plot_image_with_boxes(self, img, boxes, pred_cls, path):
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

    def cal_custom_patch_shape_and_location(self, prediction_box):
        prediction_box_width_height = (prediction_box[1][0] - prediction_box[0][0], prediction_box[1][1] - prediction_box[0][1])

        prediction_box_centre_points = (int(prediction_box[1][0] - (prediction_box_width_height[0]/2)), 
                                        int(prediction_box[1][1] - (prediction_box_width_height[1]/2))) 

        #in the format (height, width, nb_channels) to meet Dpatch Requirements
        patch_shape = (int(1/4 * prediction_box_width_height[1]), int(1/4 * prediction_box_width_height[0]), 3)
        patch_location = self.cal_custom_patch_location(prediction_box_centre_points, patch_shape)

        return patch_shape, patch_location

    def cal_custom_patch_location(self, prediction_centre_points, patch_shape): 
        #in format y,x to fit Dpatch requirements
        return ((int(prediction_centre_points[1] - (patch_shape[0]/2)),
                int(prediction_centre_points[0] - (patch_shape[1]/2)))) 

    def append_to_training_progress_file(self, string):
        path = f"{TRAINING_PROGRESS_DIRECTORY}{self.name}_training.txt"
        file_handler(path = path, mode = "a", func= lambda f: f.write("\n" + string))