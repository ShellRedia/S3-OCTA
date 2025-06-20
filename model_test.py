import torch
from funcs.model.DSCNet import DSCNet
from torchinfo import summary
import torch.nn as nn

from monai.networks.nets import *

class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.original_model(x)
        x = self.new_layer(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = SegResNet(in_channels=3, out_channels=1, spatial_dims=2, init_filters=64, blocks_down=(1, 2, 4, 4),blocks_up=(4, 4, 4))

# model = UNet(in_channels=3, out_channels=1, spatial_dims=2, channels=[512, 512, 1024, 2048, 2048], strides=[2, 2, 2, 2], kernel_size=9)

model = SwinUNETR(img_size=(512,512), in_channels=3, out_channels=1, feature_size=36, spatial_dims=2)

# dints_space = TopologyInstance(spatial_dims=2, num_blocks=12, device="cuda")
# model = DiNTS(dints_space=dints_space, in_channels=3, num_classes=1, spatial_dims=2)
model = ModifiedModel(model).to(device)

summary(model, input_size=(1,3,512,512)) 