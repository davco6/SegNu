# SegNu
SegNu is an open-software that automatically detects cell nuclei assisted by CNN.

Briefly, SegNu classify each detected object in three categories:  single cell nucleus, non-cell nucleus or cell nuclear aggregate. Afterwards, each cell nuclear aggregate is segmented recursively by watershed segmentation. Due to the fact that the watershed segmentation does not do a precisely segmentation and often get oversegmented images, each group of segmented region is considered as a graph, and each graph is optimized for merging the regions with a single cell nuclei. Eventually, SegNu detect and isolate all cell nuclei at the image.

Further information about SegNu can be found in the Master Thesis (wrote in spanish):

http://openaccess.uoc.edu/webapps/o2/handle/10609/82134

## Requirements:
* Inside the weights folder you must include the weights file, name as "model_weights.h5", generated in the training of the CNN with TensorFlow. This file could be download in:
https://drive.google.com/open?id=1Giing5bmtw81OXW2yEwxhBs2MC8PIcO-

* SegNu was written in Python 3.0, and use the following external modules:
    - numpy >=1.13.3
    - scikit-learn >= 0.13.0
    - scikit-image >= 0.19.1
    - keras >= 2.1.5
    - tensorflow >= 1.6

## Email contact:
segnuproject@gmail.com

## Important note:
This software is distributed in the hope that it will be useful for research use only, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
