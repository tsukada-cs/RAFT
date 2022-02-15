#%%
import os
from pkgutil import extend_path
import sys
sys.path.append('/Users/tsukada/git/RAFT/core')
import argparse

import cv2
import glob
import torch
import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

%load_ext autoreload
%autoreload 2

#%%
DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if img.shape[-1] >= 4:
        img = img[:,:,:3] # delete alpha cnannel
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, fname="flow.png"):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=1)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img / 255.0)
    ax[1].imshow(flo / 255.0)
    ax[0].axis("off")
    ax[1].axis("off")
    fig.savefig(fname, bbox_inches="tight", pad_inches=0.1, dpi=144,)
    plt.close()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()

def flow_to_netcdf(flow_uv, fname="flow.nc"):
    flow_uv = flow_uv[0].permute(1,2,0).cpu().numpy()
    u = flow_uv[:,:,0]
    v = -flow_uv[:,:,1] # vは画像下向き(南向き)が正なので反転
    ds = xr.Dataset(
        {"u": (['y','x'], u), "v": (['y','x'], v)},
        coords=({"y":np.r_[0:u.shape[0]],"x":np.r_[0:u.shape[1]]}))
    ds.to_netcdf(fname, mode="w")

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    # model.load_state_dict(torch.load(args.model))
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu'))) 

    model = model.module
    model.to(DEVICE)
    model.eval()

    uv_datadir = args.project+f"/results/{os.path.basename(args.model).split('.')[0]}/uv"
    os.makedirs(uv_datadir, exist_ok=True)
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        flow_nc_names = [""]*len(images)
        for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            if args.warm_start and i > 0:
                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True, flow_init=flow_low)
            else:
                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_nc_name = uv_datadir+"/"+os.path.basename(imfile1).replace(".png",".nc").replace(".jpg",".nc")
            flow_nc_names[i] = flow_nc_name
            flow_to_netcdf(flow_up, flow_nc_name)
        return images, flow_nc_names

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--project', help="project directory")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
parser.add_argument('--warm_start', action='store_true', help='use previous flow as next step initial flow')
args = parser.parse_args([
    '--model', 'models/raft-things.pth',
    '--path', '/Users/tsukada/git/RAFT/projects/test_project/frames',
    '--project', '/Users/tsukada/git/RAFT/projects/test_project',
    '--warm_start',
    ])
#%%
model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu'))) 
model = model.module
model.to(DEVICE)
model.eval()

# %%
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# %%
net = "cnet"
weights = np.array(model.state_dict()[f'{net}.conv1.weight'])
root = int(np.sqrt(weights.shape[0]))
fig, axs = plt.subplots(root, root, figsize=(8,8), facecolor="w")
for i in range(weights.shape[0]):
    w = weights[i]
    w = np.transpose(w, (1,2,0))
    wmax = w.max()
    wmin = w.min()
    w = 255 * (w-wmin)/(wmax-wmin)

    x, y = i%root, i//root
    ax = axs[y,x]
    ax.imshow(w.astype(np.uint8))
    ax.axis("off")
fig.savefig(f"{net}_weights.png", bbox_inches="tight", pad_inches=0.1)
# %%
from torch import nn 
class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3) # このレイヤーで1/2

    def forward(self, x):
        x = self.conv1(x)
        return x

conv1 = Conv()
conv1.load_state_dict({"conv1.weight": model.state_dict()[f"{net}.conv1.weight"], "conv1.bias": model.state_dict()[f"{net}.conv1.bias"]})
conv1.eval()

img = np.array(Image.open("/Users/tsukada/git/RAFT/projects/test_project/frames/test_t0.png")).astype(np.uint8)
if img.shape[-1] >= 4:
    img = img[:,:,:3] # delete alpha cnannel
img = torch.from_numpy(img).permute(2, 0, 1).float()
img = img[None]

# img = torch.tensor(img)

padder = InputPadder(img.shape) # 8の倍数に
img = padder.pad(img)[0]

img = 2 * (img / 255.0) - 1.0
img = img.contiguous()

conved = conv1(img).detach().numpy()[0,:,:,:]

weights = np.array(model.state_dict()['cnet.conv1.weight'])
root = int(np.sqrt(weights.shape[0]))
fig, axs = plt.subplots(root, root, figsize=(8,8), facecolor="w")
for i in range(weights.shape[0]):
    c = conved[i]
    x, y = i%root, i//root
    ax = axs[y,x]
    ax.imshow(c, cmap="gray")
    ax.axis("off")
fig.savefig(f"{net}_conved.png", bbox_inches="tight", pad_inches=0.1)
# %%
