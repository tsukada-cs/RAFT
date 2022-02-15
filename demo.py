import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from raft.utils import flow_viz
from raft.utils.utils import InputPadder



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
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax[0].imshow(img / 255.0)
    ax[1].imshow(flo / 255.0)
    ax[0].axis("off")
    ax[1].axis("off")
    fig.savefig(fname, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    # model.load_state_dict(torch.load(args.model))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, fname="demo-results/flo_"+os.path.basename(imfile1)+".png")
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    print(args)
    demo(args)
