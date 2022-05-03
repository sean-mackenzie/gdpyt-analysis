# gdpyt-analysis: test.test_fit_boundary
"""
Notes
"""

# imports
import numpy as np
from skimage import io, exposure
import matplotlib.pyplot as plt

from utils import boundary


# image path
fp = '/Users/mackenzie/Desktop/gdpyt-characterization/experiments/02.07.22_membrane_characterization/images/calibration/' \
     'calib_55.tif'

# read image
img = io.imread(fp)

# if image stack, take average
if len(img.shape) > 2:
    img = np.mean(img, axis=0)

# draw point and show image
x1, y1, = 10, 50
x2, y2 = 100, 50
img, mask_point = boundary.draw_boundary_points(img, x1, y1, mask=None, draw_mask=True)
img, mask_point = boundary.draw_boundary_points(img, x2, y2, mask=mask_point, draw_mask=True)
plt.imshow(img)
plt.show()

# draw circle and show image
xc, yc, r = 512, 250, 508
mask_circle = boundary.draw_boundary_circle(img, xc, yc, r)
mask_circle_rescaled = exposure.rescale_intensity(mask_circle, out_range=np.uint16)
"""
draw_on_image = True
if draw_on_image:
    img_mask = img + mask_circle_rescaled // 50
    plt.imshow(img_mask)
else:
    fig, [ax1, ax2] = plt.subplots(nrows=2)
    ax1.imshow(img)
    ax2.imshow(mask_circle_rescaled)

plt.title('circle(xc, yc, r) = ({}, {}, {})'.format(xc, yc, r))
plt.show()
"""

# apply mask to image
mask_circle_inverted = np.logical_not(mask_circle).astype(int)
img_masked = img * mask_circle_inverted
plt.imshow(img_masked)
plt.title('cirlce(xc, yc, r) = ({}, {}, {})'.format(xc, yc, r))
plt.show()

# keep particles if they are located on the boundary
boundary_indices = np.argwhere(img_masked > 0)
p1 = [x1, y1]
p2 = [x2, y2]
p3 = [y1, x1]
p4 = [250, 250]
p_coords = np.array([p1, p2, p3, p4])

passing = []
for p in p_coords:
    for b in boundary_indices:
        if all(b == p):
            passing.append(p)

h = passing


j=1