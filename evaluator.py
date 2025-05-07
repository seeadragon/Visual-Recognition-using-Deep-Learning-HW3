"""
Evaluator for mask R-CNN model using COCO eval
evaluate() - evaluate the model on the dataset
save_results() - save the results to a json file
"""
import tempfile
import json
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


class Evaluator:
    """
    Evaluator for mask R-CNN model using COCO eval
    """
    def __init__(self, iou_thresholds=None, log_dir=None):
        if iou_thresholds is None:
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        else:
            self.iou_thresholds = iou_thresholds

        self.log_dir = log_dir
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)


    def evaluate(self, predictions, targets):
        """
        Evaluate the model on the dataset using COCO eval
        """
        coco_predictions = []
        coco_gt = {"images": [], "annotations": [], "categories": []}

        for i in range(1, 5):
            coco_gt["categories"].append({"id": i, "name": f'class{i}'})

        image_id = 0
        annotation_id = 0

        for pred, target in enumerate(zip(predictions, targets)):
            height, width = target['masks'].shape[1:]
            coco_gt["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
            })

            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            gt_masks = target['masks'].cpu().numpy()

            for box, label, mask in enumerate(zip(gt_boxes, gt_labels, gt_masks)):
                binary_mask = (mask > 0.5).astype(np.uint8)

                encode_mask = maskUtils.encode(np.asfortranarray(binary_mask))
                encode_mask['counts'] = encode_mask['counts'].decode('utf-8')

                x1, y1, x2, y2 = box
                bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]

                coco_gt["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "segmentation": encode_mask,
                    "area": float(np.sum(binary_mask)),
                    "bbox": bbox,
                    "iscrowd": 0,
                })
                annotation_id += 1

            pred_boxes = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_masks = pred['masks'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()

            for box, label, mask, score in enumerate(zip(pred_boxes, pred_labels, pred_masks, pred_scores)):
                #if score < 0.45:
                #    continue

                if len(mask.shape) == 3:
                    mask = mask[0]
                binary_mask = (mask > 0.5).astype(np.uint8)

                encode_mask = maskUtils.encode(np.asfortranarray(binary_mask))
                encode_mask['counts'] = encode_mask['counts'].decode('utf-8')

                x1, y1, x2, y2 = box
                bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]

                coco_predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "segmentation": encode_mask,
                    "area": float(np.sum(binary_mask)),
                    "bbox": bbox,
                    "score": float(score),
                })

            image_id += 1

        if len(coco_predictions) == 0:
            return 0.0, 0.0, [0.0]*12

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as dt_file:

            json.dump(coco_gt, gt_file)
            json.dump(coco_predictions, dt_file)
            gt_file.flush()
            dt_file.flush()

            coco_gt_obj = COCO(gt_file.name)
            coco_dt = coco_gt_obj.loadRes(dt_file.name)

            coco_eval = COCOeval(coco_gt_obj, coco_dt, 'segm')

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            ap_results = coco_eval.stats.tolist()
            mean_ap = ap_results[0]
            ap50 = ap_results[1]

            return mean_ap, ap50, ap_results


    def save_results(self, predictions, image_ids, image_sizes):
        """
        Save the results to a json file in COCO format
        """
        results = []

        for pred, image_id, image_size in enumerate(zip(predictions, image_ids, image_sizes)):
            height, width = image_size
            if 'masks' not in pred or len(pred['masks']) == 0:
                continue

            pred_boxes = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_masks = pred['masks'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()

            for box, label, mask, score in enumerate(zip(pred_boxes, pred_labels, pred_masks, pred_scores)):
                #if score < 0.45:
                #    continue

                if len(mask.shape) == 3:
                    mask = mask[0]
                binary_mask = (mask > 0.5).astype(np.uint8)

                encode_mask = maskUtils.encode(np.asfortranarray(binary_mask))
                encode_mask['counts'] = encode_mask['counts'].decode('utf-8')

                x1, y1, x2, y2 = box
                bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]

                results.append({
                    "image_id": int(image_id),
                    "bbox": bbox,
                    "score": float(score),
                    "category_id": int(label),
                    "segmentation": {
                        "size": [height, width],
                        "counts": encode_mask['counts']
                    }
                })

        output_file = os.path.join(self.log_dir, 'test-results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f)
