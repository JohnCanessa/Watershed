# **** imports ****
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import data, util, filters, color

#from skimage.morphology import watershed
from skimage.segmentation import watershed


# **** read bubbles image as grayscale ****
bubbles = color.rgb2gray(io.imread('./images/pexels-bubbles.jpg'))

# **** display the grayscale bubbles image ****
plt.figure(figsize=(10, 10))
plt.imshow(bubbles, cmap='gray')
plt.title('Bubbles')
plt.show()


# **** Apply the sobel filter to the bubbles image.
#      A watershed algorithm will perform better if we use
#      an edge detected image. ****
bubbles_edges = filters.sobel(bubbles)

# **** display the edges of the bubbles image ****
plt.figure(figsize=(10, 10))
plt.imshow(bubbles_edges, cmap='gray')
plt.title('Bubbles Edges')
plt.show()


# **** Find 300 points regularly spaced along the image ****
grid = util.regular_grid(   bubbles.shape, 
                            n_points=300)

# **** The points are returned in the form of slices-
#      one slice for each dimension ****
grid
[slice(15, None, 30), slice(15, None, 30)]

# **** The seeds matrix is the same shape as the
#      image and contains integers ranging
#      from 1 to size of image ****
seeds = np.zeros(bubbles.shape, dtype=int)
seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1

# **** Seeds are the image markers from where
#      the flooding should begin.
#      The classic watershed algorithm may produce uneven
#      fragments - maybe hard to perform further analysis. ****
w0 = watershed( bubbles_edges, 
                seeds)
water_classic = color.label2rgb(w0, 
                                bubbles,
                                alpha=0.4,
                                kind='overlay')

# **** display the classic watershed image ****
plt.figure(figsize=(8, 8))
plt.imshow(water_classic)
plt.title('Classic Watershed')
plt.show()


# **** Run compact watershed algorithm.
#      Any compactness value > 0 produces a
#      compact watershed - produces even fragments. ****
w1 = watershed( bubbles_edges, 
                seeds, 
                compactness=0.91)

water_compact = color.label2rgb(w1,
                                bubbles,
                                alpha=0.4,
                                kind='overlay')

# **** Display the compact watershed image ****
plt.figure(figsize=(8, 8))
plt.imshow(water_compact)
plt.title('Compact Watershed')
plt.show()
