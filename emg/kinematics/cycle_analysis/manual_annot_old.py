# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
# ---

# %%
import time

import matplotlib.pyplot as plt
import numpy as np


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


plt.clf()
plt.setp(plt.gca(), autoscale_on=False)

tellme("You will define a triangle, click to begin")

plt.waitforbuttonpress()

while True:
    pts = []
    while len(pts) < 3:
        tellme("Select 3 corners with mouse")
        pts = np.asarray(plt.ginput(3, timeout=-1))
        if len(pts) < 3:
            tellme("Too few points, starting over")
            time.sleep(1)
    ph = plt.fill(pts[:, 0], pts[:, 1], "r", lw=2)

    tellme("Happy? Key click for yes, mouse click for no")

    if plt.waitforbuttonpress():
        break

    # Get rid of fill
    for p in ph:
        p.remove()


plt.show()
print(pts)
