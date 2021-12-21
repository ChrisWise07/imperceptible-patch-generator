import torch

from typing import List
from dataclasses import dataclass, field
from torch.functional import Tensor
from utils import generate_predictions, open_image_as_rgb_np_array
from constants import FRCNN, EPSILON, LIMIT_OF_PREDICTIONS_PER_IMAGE, DEVICE

#adapted from: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	inter_rectangle_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	return inter_rectangle_area / float(boxAArea + boxBArea - inter_rectangle_area)
	
@dataclass(repr=False, eq=False)
class mAP_calculator:
	confidence_threshold: float
	number_of_images: int
	counter: int = 0
	mAP: float = 0.0
	unsorted_confidence_values: Tensor = field(init=False)
	unsorted_tp: Tensor = field(init=False)
	unsorted_fp: Tensor = field(init=False)

	def __post_init__(self):
		self.unsorted_confidence_values = torch.zeros(self.number_of_images * LIMIT_OF_PREDICTIONS_PER_IMAGE, dtype=torch.float, device=DEVICE)
		self.unsorted_tp = torch.zeros(self.number_of_images * LIMIT_OF_PREDICTIONS_PER_IMAGE, dtype=torch.short, device=DEVICE)
		self.unsorted_fp = torch.zeros(self.number_of_images * LIMIT_OF_PREDICTIONS_PER_IMAGE, dtype=torch.short, device=DEVICE)

	def map_confidence_to_tp_fp(self, ground_truth_bbox: List[tuple], adv_image_path: str):
		predictions_class, predictions_boxes, predictions_score = generate_predictions(object_detector=FRCNN, 
																					   image=open_image_as_rgb_np_array(path=adv_image_path), 
																					   threshold=self.confidence_threshold)
		best_iou, best_iou_index = 0.5, 0

		for pred_class, bbox, score in zip(predictions_class, predictions_boxes, predictions_score):
			self.unsorted_confidence_values[self.counter] = float(score)
			if (pred_class == "airplane"):
				iou = bb_intersection_over_union([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]], ground_truth_bbox)
				if (iou > best_iou):
					best_iou = iou
					best_iou_index = self.counter
				else:
					self.unsorted_fp[self.counter] = 1
			else:
				self.unsorted_fp[self.counter] = 1

			self.counter += 1

		self.unsorted_tp[best_iou_index] = 1

	def calculate_mAP(self) -> float:
		sorted_confidence_values, indices = torch.sort(self.unsorted_confidence_values[:self.counter], descending=True)

		self.unsorted_tp = self.unsorted_tp[:self.counter]
		self.unsorted_fp = self.unsorted_fp[:self.counter]
		sorted_tp = torch.tensor([self.unsorted_tp[i] for i in indices], dtype=torch.short, device=DEVICE)
		sorted_fp = torch.tensor([self.unsorted_fp[i] for i in indices], dtype=torch.short, device=DEVICE)
		tp_cum_sum = torch.cumsum(sorted_tp, dim=0, dtype=torch.short)
		fp_cum_sum = torch.cumsum(sorted_fp, dim=0, dtype=torch.short)
		precisions = torch.divide(tp_cum_sum, (tp_cum_sum + fp_cum_sum + EPSILON))
		precisions = torch.cat((torch.tensor([1], dtype=torch.float, device=DEVICE), precisions))
		recalls = torch.divide(tp_cum_sum, (self.number_of_images + EPSILON))
		recalls = torch.cat((torch.tensor([0], dtype=torch.float, device=DEVICE), recalls))
		
		self.mAP = torch.trapz(precisions, recalls).item()
		return self.mAP 