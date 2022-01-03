#%%
import os
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


def create_img(x, y, r, color=(0,0,0)):
    img = np.full([200,200,3], 255, np.uint8)
    cv2.circle(img, (x, y), r, color, thickness=-1)
    return img

for t in range(5):
    img = create_img(10*t**2+20, 5*t**2+20, 10, color=(150,20,150))
    img[25:175, 90:110, :] = 100
    cv2.imwrite(f'projects/test_project/frames/test_t{t}.png', img)
    plt.imshow(img)
    plt.show()
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
    '--path', 'projects/202010_Haishen/frames',
    '--project', 'projects/202010_Haishen',
    '--warm_start',
    ])
#%%
imgs, flows = demo(args)

#%%
i = 0
img = load_image(imgs[i])
img = img[0].permute(1,2,0).cpu().numpy()

flow = xr.open_dataset(flows[i])
u, v = flow.u.values, flow.v.values

fig, ax = plt.subplots(figsize=(8,8))
mabiki = 8
u_view = u[::mabiki,::mabiki]
v_view = v[::mabiki,::mabiki]
velocity = np.hypot(u_view, v_view)
yy, xx = np.mgrid[0:u.shape[0], 0:u.shape[1]]
rr = np.hypot(xx-np.median(xx), yy-np.median(yy))
ax.imshow(img/255.0, origin="upper")
ax.quiver(xx[::mabiki,::mabiki], yy[::mabiki,::mabiki], u_view, v_view, velocity, scale=img.shape[0], cmap="jet")
ax.axis("off")
ax.set(aspect="equal")

out_dir = f"{args.project}/results/{os.path.basename(args.model).split('.')[0]}/flow_viz"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f"{out_dir}/flow_{os.path.basename(imgs[i])}", bbox_inches="tight", pad_inches=0, dpi=200)
# %%
fig, ax = plt.subplots(1, 2, figsize=(8,8))
velocities = np.hypot(u, v)
ax[0].imshow(velocities, cmap="jet")
directions = np.arctan2(v, u)
directions[velocities < 1] = 0
ax[1].imshow(directions, cmap="jet")

# %%