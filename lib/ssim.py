"""A class for calculating structural similarity index measure (SSIM), 
with added functionality to ignore NaN values in the target image.

Adapted from https://github.com/Po-Hsun-Su/pytorch-ssim"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, filter_nan=False):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    if filter_nan:
        mask = torch.isnan(mu2)
        img2 = torch.where(torch.isnan(img2), torch.zeros_like(img2), img2)
        mu2 = torch.where(mask, torch.zeros_like(mu2), mu2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    # Filter out NaN values
    if size_average:
        if filter_nan:
            return ssim_map[~mask].sum()/(~mask).sum()
        else:
            return ssim_map.mean()
    else:
        if filter_nan:
            raise ValueError("not implemented")
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=1, device='cuda', size_average=True, filter_nan=False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.filter_nan = filter_nan
        self.window = create_window(window_size, self.channel).to(device)

    def forward(self, img1, img2):
        return _ssim(img1, img2, self.window, self.window_size, self.channel,
                                self.size_average, self.filter_nan)


if __name__=='__main__':
    # example usage
    # 3 channels for RGB
    # device = 'cuda' if using GPU else device='cpu'
    calc_ssim = SSIM(channel=3, device='cuda')

    # loss function is negative SSIM
    loss_fn = lambda y, y_pred: -calc_ssim(y, y_pred)

    # fake images. N real images and N predicted images
    # 3 channels and 127x127 pixels
    N = 10
    y = torch.rand((N, 3, 127, 127))
    y_pred = torch.rand((N, 3, 127, 127))

    # calculate losss
    loss = loss_fn(y, y_pred)
