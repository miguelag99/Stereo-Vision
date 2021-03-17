import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


#def draw_2D_bbox(img,):


def main():
    path = []

    path.append("/home/miguel/TFG/Stereo-Vision/Datasets/image_left/")
    path.append("/home/miguel/TFG/Stereo-Vision/Datasets/label_2/")
    path.append("/home/miguel/TFG/Stereo-Vision/Datasets/camera_param/calib/")

    i = 250

    image_file = sorted(os.listdir(path[0]))[i]
    label_file = sorted(os.listdir(path[1]))[i]
    param = sorted(os.listdir(path[2]))[i]

    imgL = cv2.imread(path[0]+image_file,1)
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    assert(imgL is not None)

    camera_param = open(path[2]+param,'r')
    matrix = camera_param.readlines()
    matrix = (matrix[2].split(':'))[1].split(' ')
    print(matrix)
    Labels = open(path[1]+label_file,'r')
    camera_param = open()
    for label in Labels.readlines():
        field = label.split(' ')
        if field[0] != 'DontCare':
            point1 = (int(float(field[4])),int(float(field[5])))
            point2 = (int(float(field[6])),int(float(field[7])))
            center_world = (int(float(field[11])),int(float(field[12])),int(float(field[13])))
            cv2.rectangle(imgL,point1,point2,(0,255,0),1)
            #cv2.circle(imgL,center,100,(0,0,255))
        
    Labels.close()

    #imgR = cv2.imread(path[1]+image,0)
    #assert(imgR is not None)

    #stereo = cv2.StereoBM_create(numDisparities=112, blockSize=39)
    #disparity = stereo.compute(imgL,imgR)

    #plt.imshow(disparity,'gray')
    
    plt.imshow(imgL)
    plt.show()





if __name__ == "__main__":
    main()
