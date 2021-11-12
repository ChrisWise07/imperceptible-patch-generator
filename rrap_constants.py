import os
import sys
import torch

from torchvision import transforms
from rrap_custom_pytorch_faster_rcnn import PyTorchFasterRCNN

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
sys.path.append(ROOT_DIRECTORY)

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

TRANSFORM = transforms.Compose([transforms.ToTensor(),])

FRCNN = PyTorchFasterRCNN(
        clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    )

if not torch.cuda.is_available():
        DEVICE = torch.device("cpu")
else:
        cuda_idx = torch.cuda.current_device()
        DEVICE = torch.device(f"cuda:{cuda_idx}")

DATA_DIRECTORY = ROOT_DIRECTORY + "/data/"

PATCHED_IMAGE_PATH = DATA_DIRECTORY + "temp/patch_image.jpg"

INITIAL_PREDICTIONS_DIRECTORY = DATA_DIRECTORY + "initial_predictions/"

IMAGES_DIRECTORY = DATA_DIRECTORY + "images/"

PATCHES_DIRECTORY = DATA_DIRECTORY + "patches_adv/"

ADVERSARIAL_PREDICTIONS_DIRECTORY = DATA_DIRECTORY + "predictions_adv/"

ADVERSARIAL_IMAGES_DIRECTORY =  DATA_DIRECTORY + "images_adv/"

TRAINING_DATA_DIRECTORY =  DATA_DIRECTORY + "attack_training_data/"

TRAINING_PROGRESS_DIRECTORY = DATA_DIRECTORY + "training_progress_data/"

PLOTS_DIRECTORY = DATA_DIRECTORY + "plots/"