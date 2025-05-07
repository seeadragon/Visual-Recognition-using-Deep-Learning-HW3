"""
New Mask Predictor
3 layers of conv2d with ReLU
"""
from torch import nn

class MaskPredictor(nn.Module):
    """
    new Mask Predictor
    self.model.roi_heads.mask_predictor = MaskPredictor(in_channels, hidden_layer, num_classes)
    """
    def __init__(self, in_channels, hidden_layer, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_layer, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_layer, hidden_layer, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(hidden_layer, hidden_layer, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.mask_fcn_logits = nn.Conv2d(hidden_layer, num_classes, 1, 1, 0)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.mask_fcn_logits(x)
        return x
