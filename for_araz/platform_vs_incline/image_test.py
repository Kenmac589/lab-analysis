import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = np.asarray(
    Image.open(
        "../../msc-thesis-related/thesis_figures/DTR-M6-predtx-000002-right-ds.jpg"
    )
)

print(repr(img))

imgplot = plt.imshow(img)
plt.axis("off")
plt.show()
