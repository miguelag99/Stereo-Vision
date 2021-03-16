import numpy as np
import cv2
from matplotlib import pyplot as plt

path = "/home/miguel/TFG/Stereo-Vision/"

imgL = cv2.imread(path+"/Datasets/000000.png",-1)
imgR = cv2.imread(path+"/Datasets/000000_R.png",-1)
assert(imgL is not None)
assert(imgR is not None)

#plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()