import torch.nn as nn
import torch

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, img_shape=(64, 64, 3), latent_dim=256):
        super(Generator, self).__init__()

        h, w, c = img_shape
        self.img_shape = img_shape

        self.ll = nn.Linear(latent_dim, 256*49)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=32, out_channels=c, kernel_size=4, stride=2, padding=1, bias=False),

            nn.Tanh(),
        )

    def forward(self, z):
        out = self.ll(z)
        out = out.view(out.shape[0], 256, 7, 7)
        img = self.conv_blocks(out)
        return img


    
