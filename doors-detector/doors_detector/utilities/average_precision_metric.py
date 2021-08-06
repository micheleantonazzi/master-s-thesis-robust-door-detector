from src.bounding_box import BoundingBox


class AveragePrecisionCalculator:
    def __init__(self):
        self._gt_bboxes = []
        self._predicted_bboxes = []

        self._img_count = 0

    def add_prediction(self, ground_truth, prediction):
        print(prediction)
        print(ground_truth)

