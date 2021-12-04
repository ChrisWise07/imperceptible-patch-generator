from typing import List
from rrap_utils import generate_predictions, open_image_as_rgb_np_array
from rrap_constants import FRCNN

#sourced from: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou
    

def calculate_mAP(ground_truth_bbox: List[tuple], adv_image_path: str):
    predictions_class, predictions_boxes, predictions_score = generate_predictions(object_detector=FRCNN, 
                                                                                   image=open_image_as_rgb_np_array(path=adv_image_path), 
                                                                                   threshold=0.5)
    for prediction_bbox in predictions_boxes:
        x1, y1, x2, y2 = prediction_bbox[0][0], prediction_bbox[0][1], prediction_bbox[1][0], prediction_bbox[1][1]
        print(bb_intersection_over_union([x1, y1, x2, y2], ground_truth_bbox))
