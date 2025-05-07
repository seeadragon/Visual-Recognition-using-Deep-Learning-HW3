"""
Mask R-CNN model
train_model() - train the model on the training set and validate on the validation set
valid() - validate the model on the validation set
test() - test the model on the test set and save results in json

load_model() - load model checkpoint
count_params() - count the number of parameters in the model
"""
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNN as MaskRCNNBase
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from dataloader import GetLoader
from evaluator import Evaluator
from mask_predictor import MaskPredictor

class MaskRCNN(nn.Module):
    """
    backbone: ResNet50 with FPN
    Anchor: sizes=(4, 8, 16, 32, 64), aspect_ratios=(0.5, 1.0, 2.0)
    roi_pooler: MultiScaleRoIAlign
    mask_roi_pooler: MultiScaleRoIAlign
    box_predictor: FastRCNNPredictor
    mask_predictor: MaskRCNNPredictor
    mask_predictor: MaskPredictor 3 conv layers
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.num_classes = config['num_classes']
        self.device = config['device']
        self.epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.optimizer = config['optimizer']
        self.log_dir = config['log_dir']
        self.weight_decay = config['weight_decay']

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        backbone_with_fpn = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=ResNet50_Weights.IMAGENET1K_V2,
            trainable_layers=5,
        )

        anchor_generator = AnchorGenerator(
            sizes=((4,), (8,), (16,), (32,), (64,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        mask_roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=14,
            sampling_ratio=4
        )
        self.model = MaskRCNNBase(
            backbone=backbone_with_fpn,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            mask_roi_pool=mask_roi_pooler
        )
        self.model.roi_heads.positive_fraction = 0.25
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            self.num_classes
        )

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            self.num_classes
        )
        ### new mask predictor
        #self.model.roi_heads.mask_predictor = MaskPredictor(
        #    in_features_mask,
        #    hidden_layer,
        #    self.num_classes
        #)


    def forward(self, x, targets=None):
        """ forward """
        if self.training and targets is not None:
            return self.model(x, targets)
        return self.model(x)

    def predict(self, x):
        """ predict """
        self.eval()
        with torch.no_grad():
            predictions = self.model(x)
        return predictions


    def train_model(self):
        """
        training
        """
        self.model.train()
        self.model.to(self.device)

        data_loader = GetLoader(data_dir='data')
        train_loader, _ = data_loader.train_loader()
        valid_loader, _ = data_loader.valid_loader()

        evaluator = Evaluator(log_dir=self.log_dir)

        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.75,
            patience=3,
            min_lr=1e-6
        )

        best_map = 0.0
        counter = 1

        for epoch in range(self.epochs):
            torch.cuda.empty_cache()

            train_loss = self._train_epoch(train_loader, optimizer)
            self.writer.add_scalar('train/loss', train_loss, epoch)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}")

            val_loss, mean_ap, ap50, _  = self._eval_epoch(valid_loader, evaluator)
            self.writer.add_scalar('valid/loss', val_loss, epoch)
            self.writer.add_scalar('valid/mAP', mean_ap, epoch)
            self.writer.add_scalar('valid/AP50', ap50, epoch)
            print(f"Epoch {epoch+1}/{self.epochs}, Validation Loss: {val_loss:.4f}, mAP: {mean_ap:.4f}, AP50: {ap50:.4f}")

            if mean_ap > best_map:
                best_map = mean_ap
                model_path = os.path.join(self.log_dir, "best_model.pth")
                self.save_model(model_path)

            if (epoch+1) % 5 == 0:
                model_path = os.path.join(self.log_dir, f"model_{counter}_map_{mean_ap:.3f}.pth")
                self.save_model(model_path)
                counter += 1
            lr_scheduler.step(mean_ap)


        self.writer.close()

    def valid(self):
        """
        validation
        """
        self.model.to(self.device)
        self.model.eval()
        data_loader = GetLoader(data_dir='data')
        valid_loader, _ = data_loader.valid_loader()

        evaluator = Evaluator(log_dir=self.log_dir)
        val_loss, mean_ap, ap50, ap_results  = self._eval_epoch(valid_loader, evaluator)

        print(f"valid: , Validation Loss: {val_loss:.4f}, mAP: {mean_ap:.4f}, AP50: {ap50:.4f}")
        print(f"AP results: {ap_results}")

    def test(self):
        """
        test set and save results in json
        """
        test_loader, _ = GetLoader(data_dir='data').test_loader()
        self.model.to(self.device)
        self.model.eval()

        evaluator = Evaluator(log_dir=self.log_dir)
        all_predictions = []
        all_image_ids = []
        all_image_sizes = []

        with torch.no_grad():
            with tqdm(test_loader, desc="Test") as pbar:
                for batch in pbar:
                    images, image_ids, heights, widths = batch
                    images_list = [img.to(self.device) for img in images]

                    predictions = self.model(images_list)
                    for i, pred in enumerate(predictions):
                        all_predictions.append(pred)
                        all_image_ids.append(image_ids[i])
                        all_image_sizes.append((heights[i], widths[i]))

                    pbar.set_postfix({"image_id": all_image_ids[-1]})

        evaluator.save_results(all_predictions, all_image_ids, all_image_sizes)
        print("Results saved")


    def _train_epoch(self, train_loader, optimizer):
        """
        train the model on the training set 1 epoch
        """
        self.model.train()
        train_loss = 0.0
        counter = 0
        with tqdm(train_loader, desc="Train") as pbar:
            for images, targets in pbar:
                torch.cuda.empty_cache()
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                train_loss += losses.item()
                counter += 1
                pbar.set_postfix(loss=losses.item())

        avg_loss = train_loss / counter if counter > 0 else 0
        return avg_loss


    def _eval_epoch(self, valid_loader, evaluator):
        """
        evaluate the model on the validation set 1 epoch
        """
        self.model.eval()
        val_loss = 0.0
        counter = 0

        all_predictions = []
        all_targets = []
        with torch.no_grad():
            with tqdm(valid_loader, desc="Valid") as pbar:
                for images, targets in pbar:
                    torch.cuda.empty_cache()

                    predictions = []
                    for i, item in enumerate(images):
                        image = [item.to(self.device)]
                        target = [{k: v.to(self.device) for k, v in targets[i].items()}]

                        pred = self.model(image)
                        predictions.extend(pred)
                        torch.cuda.empty_cache()

                    all_predictions.extend(predictions)
                    all_targets.extend(targets)

                    loss_value = 0
                    for i, item in enumerate(images):
                        self.model.train()
                        image = [item.to(self.device)]
                        target = [{k: v.to(self.device) for k, v in targets[i].items()}]
                        loss_dict = self.model(image, target)
                        self.model.eval()

                        if isinstance(loss_dict, dict):
                            losses = sum(loss for loss in loss_dict.values())
                            loss_value += losses.item()

                    val_loss += loss_value
                    counter += 1
                    pbar.set_postfix(loss=loss_value/len(images))
                    del images, targets, predictions, loss_dict, losses
                    torch.cuda.empty_cache()

        avg_loss = val_loss / counter if counter > 0 else 0
        mean_ap, ap50, ap_results = evaluator.evaluate(all_predictions, all_targets)

        return avg_loss, mean_ap, ap50, ap_results


    def load_model(self, model_path):
        """
        load model checkpoint
        """
        self.model.load_state_dict(torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        ))
        self.model.to(self.device)


    def save_model(self, model_path):
        """
        save model checkpoint
        """
        torch.save(self.model.state_dict(), model_path)


    def count_params(self):
        """
        Returns:
            total params: int
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params
