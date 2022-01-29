import torch

from typing import Dict, List, Tuple
from dataclasses import dataclass
from torch.functional import Tensor
from utils import generate_predictions, open_image_as_rgb_np_array
from constants import FRCNN, EPSILON, LIMIT_OF_PREDICTIONS_PER_IMAGE, DEVICE

#adapted from:
#https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
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

	def single_image_map_confidence_to_tp_fp(
			self, ground_truth_bbox: List[float], adv_image_path: str, 
			unsorted_confidence_values: Tensor, unsorted_tp: Tensor, 
			unsorted_fp: Tensor) -> None:
			
		predictions_class, predictions_boxes, predictions_score = generate_predictions(
			object_detector=FRCNN, 
			image=open_image_as_rgb_np_array(path=adv_image_path), 
			threshold=self.confidence_threshold
		)

		best_iou, best_iou_index = 0.5, 0

		for pred_class, bbox, score in zip(
			predictions_class, predictions_boxes, predictions_score
		):
			unsorted_confidence_values[self.counter] = float(score)

			if (pred_class == "airplane"):
				iou = bb_intersection_over_union(
					[bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]], 
					ground_truth_bbox
				)

				if (iou > best_iou):
					best_iou = iou
					best_iou_index = self.counter
				else:
					unsorted_fp[self.counter] = 1

			else:
				unsorted_fp[self.counter] = 1

			self.counter += 1

		unsorted_tp[best_iou_index] = 1
	
	def sort_tp_fp_by_confidence(
			self, unsorted_confidence_values: Tensor, 
			unsorted_tp: Tensor, unsorted_fp: Tensor) -> Tuple[Tensor, Tensor]:

		sorted_confidence_values, indices = torch.sort(
			unsorted_confidence_values[:self.counter], descending=True
		)

		unsorted_tp = unsorted_tp[:self.counter]
		unsorted_fp = unsorted_fp[:self.counter]

		sorted_tp = torch.tensor(
			[unsorted_tp[i] for i in indices], dtype=torch.short, device=DEVICE
		)

		sorted_fp = torch.tensor(
			[unsorted_fp[i] for i in indices], dtype=torch.short, device=DEVICE
		)

		return sorted_tp, sorted_fp

	def calculate_area_under_curve(self, sorted_tp: Tensor, sorted_fp: Tensor) -> None:
		tp_cum_sum = torch.cumsum(sorted_tp, dim=0, dtype=torch.short)
		fp_cum_sum = torch.cumsum(sorted_fp, dim=0, dtype=torch.short)
		precisions = torch.divide(tp_cum_sum, (tp_cum_sum + fp_cum_sum + EPSILON))
		precisions = torch.cat((torch.tensor([1], dtype=torch.float, device=DEVICE), precisions))
		recalls = torch.divide(tp_cum_sum, (self.number_of_images + EPSILON))
		recalls = torch.cat((torch.tensor([0], dtype=torch.float, device=DEVICE), recalls))
		
		self.mAP = torch.trapz(precisions, recalls).item()

	def calculate_mAP(
			self, 
			ground_truths: Dict[str, List[float]], 
			file_name_type: List[List[str]]) -> None:

		from main import final_patched_images_directory
			
		unsorted_confidence_values = torch.zeros(
			self.number_of_images * LIMIT_OF_PREDICTIONS_PER_IMAGE, 
			dtype=torch.float, device=DEVICE
		)
		unsorted_tp = torch.zeros(
			self.number_of_images * LIMIT_OF_PREDICTIONS_PER_IMAGE, 
			dtype=torch.short, device=DEVICE
		)
		unsorted_fp = torch.zeros(
			self.number_of_images * LIMIT_OF_PREDICTIONS_PER_IMAGE, 
			dtype=torch.short, device=DEVICE
		)

		for name, type in file_name_type:
			self.single_image_map_confidence_to_tp_fp(
				ground_truths[name], f"{final_patched_images_directory}/adv_{name}.{type}", 
				unsorted_confidence_values, unsorted_tp, unsorted_fp
			)

		sorted_tp, sorted_fp = self.sort_tp_fp_by_confidence(
			unsorted_confidence_values, unsorted_tp, unsorted_fp
		)

		self.calculate_area_under_curve(sorted_tp, sorted_fp)