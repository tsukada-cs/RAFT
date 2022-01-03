#%%
import sys
sys.path.append('/Users/tsukada/git/RAFT/core')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import flow_viz


# %%
u, v = np.mgrid[-1:1:1e-2, -1:1:1e-2]
flow = np.stack([u,v]).T
colors = flow_viz.flow_to_image(flow)
# %%
fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(colors)
patch = mpatches.Circle((100,100), radius=99, transform=ax.transData)
im.set_clip_path(patch)
ax.axis("off")
fig.savefig("/Users/tsukada/git/RAFT/etc/color_wheel.png", bbox_inches="tight", pad_inches=0, dpi=300)
# %%