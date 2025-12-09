# model.py
import torch
import torch.nn as nn
import torchvision

class CSRNet(nn.Module):
    def __init__(self, load_vgg_pretrained=True):
        super(CSRNet, self).__init__()

        # --- Frontend: use pretrained VGG16 features ---
        # We will take the first 23 layers of VGG16 features (conv layers up to conv4_3)
        # This matches common CSRNet implementations.
        vgg16 = torchvision.models.vgg16(pretrained=load_vgg_pretrained)
        vgg_features = list(vgg16.features.children())
        self.frontend = nn.Sequential(*vgg_features[:23])  # conv1..conv4_3

        # --- Backend: dilated convolution layers ---
        # Common CSRNet backend configuration: dilation=2 to increase receptive field
        backend_cfg = [512, 512, 512, 256, 128, 64]
        backend_layers = []
        in_channels = 512  # frontend output channels
        for out_channels in backend_cfg:
            backend_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels

        self.backend = nn.Sequential(*backend_layers)

        # --- Final output layer: single-channel density map ---
        self.output_layer = nn.Conv2d(in_channels, 1, kernel_size=1)

        # Initialize backend and output layer weights
        self._initialize_weights()

    def forward(self, x):
        """
        x: input image tensor, shape (B, 3, H, W), float in [0,1] or normalized similarly.
        returns: density map tensor, shape (B, 1, H_out, W_out)
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        # Initialize backend and output layer using kaiming normal (common practice)
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # output layer init
        m = self.output_layer
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
