from typing import Dict, List

import numpy as np
from src.bounding_box import BoundingBox

from doors_detector.evaluators.model_evaluator import ModelEvaluator


class MyEvaluator(ModelEvaluator):

    def get_metrics(self, iou_threshold: float = 0.5, confidence_threshold: float = 0.5) -> Dict:
        """
        This method calculates metrics to evaluate a object detection model.
        This metric is develop specifically for a robotic context.
        In fact, if the model is used by a robot, it has to process a lot of negative images (images without any object to detect).
        To correctly evaluate the model's performance in a robotic context, it is mandatory to consider also the negative images.
        This metric works as follow:
        1) a new label is introduced: -1. It indicates the negative images
        2) the predicted bounding boxes are filtered by their confidence using confidence_threshold parameter.
            Each bbox with confidence < confidence_threshold is discarded
        3) For each positive image, the bounding boxes are processed as follow.
            Each predicted bbox is matched with a single ground truth bounding box. If a ground truth bbox has not a match, it is considered a FN,
            while a predicted bounding box with no match is considered a FP.
            A match is composed by a ground truth bbox and a predicted bbox: they must have the same label and the iou values grater than the other predictions.
            Every matched predicted bbox is a TP.
        4) For each negative image, a new label is added (-1), it indicates the negative images with no predictions
            (all predicted bounding boxes have confidence < confidence_threshold)
            If some predictions have an higher confidence, they are considered as FP

        TP = ground predicted bbox that can be matched with a ground truth one or negative image with no predictions
        FP = a predicted bounding with no match (because the iuo < iou_threshold or the correspondent gt_bbox has already been matched)
        FN = a ground truth bounding box with no match or the correspondent predicted box has the wrong label
        :param iou_threshold:
        :param confidence_threshold:
        :return:
        """
        gt_bboxes = self.get_gt_bboxes()
        predicted_bboxes = self.get_predicted_bboxes()

        # A dictionary containing all bboxes divided by image. DETR produces a fixed number of prediction for every image.
        # This means that also the negative images are considered using the predicted_bboxes list
        bboxes_images = {
            box.get_image_name(): {
                'gt_bboxes': [],
                'predicted_bboxes': [],
                'TP': 0,
                'FP': 0,
            }
            for box in predicted_bboxes}

        # Add bboxes to each image
        [bboxes_images[box.get_image_name()]['gt_bboxes'].append(box) for box in gt_bboxes]
        [bboxes_images[box.get_image_name()]['predicted_bboxes'].append(box) for box in predicted_bboxes if box.get_confidence() >= confidence_threshold]

        # Create information by label
        labels = set([box.get_class_id() for box in gt_bboxes])
        labels.add('-1')

        result_by_labels = {
            label: {
                'total_positives': 0,
                'TP': [],
                'FP': [],
            } for label in labels
        }

        for img, values in bboxes_images.items():
            gt_bboxes: List[BoundingBox] = values['gt_bboxes']
            predicted_bboxes: List[BoundingBox] = values['predicted_bboxes']

            # Update total positives
            for gt_box in gt_bboxes:
                result_by_labels[gt_box.get_class_id()]['total_positives'] += 1

            if len(gt_bboxes) == 0:
                # Negative images
                pass
            else:
                # Positive images

                gt_mask = np.zeros(len(gt_bboxes))

                # Match ground truth and predicted bboxes
                for p_index, p_box in enumerate(predicted_bboxes):
                    p_label = p_box.get_class_id()
                    iou_max = float('-inf')
                    match_index = -1

                    # Find the grater iou area with gt bboxes
                    for gt_index, gt_box in enumerate(gt_bboxes):
                        iou = BoundingBox.iou(p_box, gt_box)
                        if iou > iou_max:
                            iou_max = iou
                            match_index = gt_index

                    # If the iou >= threshold_iou and the label is the same, the match is valid
                    if iou_max >= iou_threshold:
                        # True Positive
                        if gt_mask[match_index] == 0 and gt_bboxes[match_index].get_class_id() == p_label:
                            # Set gt bbox as matched
                            gt_mask[match_index] = 1

                            # Update image information
                            values['TP'] += 1

                            # Update label information
                            result_by_labels[p_label]['TP'].append(1)
                            result_by_labels[p_label]['FP'].append(0)
                        # False Positive (if the gt box has already been matched or the label is different)
                        else:
                            # Update image information
                            values['FP'] += 1

                            # Update label information
                            result_by_labels[p_label]['TP'].append(0)
                            result_by_labels[p_label]['FP'].append(1)
                    # False Positive (iou < iou threshold)
                    else:
                        # Update image information
                        values['FP'] += 1

                        # Update label information
                        result_by_labels[p_label]['TP'].append(0)
                        result_by_labels[p_label]['FP'].append(1)

        labels_information = {}

        for label, values in result_by_labels.items():
            ret = {
                'total_positives': values['total_positives'],
                'TP': np.count_nonzero(values['TP']),
                'FP': np.count_nonzero(values['FP'])
            }
            labels_information[label] = ret

        return labels_information
