import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import math

def read_params(file_path):

    camera_param = open(file_path,'r')
    matrix = camera_param.readlines()
    camera_param.close()
    matrix = (matrix[2].split(':'))[1].split(' ')
    matrix.pop(0)
    matrix[11] = matrix[11].rstrip('\n')
    matrix = [float(i) for i in matrix]
    
    p = np.vstack((matrix[0:4],matrix[4:8],matrix[8:12]))
    return p


def print2D_centers(imgL,label_path,p):
    labels = open(label_path,'r')
    for label in labels:
        field = label.split(' ')
        if field[0] != 'DontCare':
            point1 = (int(float(field[4])),int(float(field[5])))
            point2 = (int(float(field[6])),int(float(field[7])))
            cv2.rectangle(imgL,point1,point2,(0,255,0),1)
            center = field[11:14]
            center = [float(i) for i in center]
            center.append(1)
            center = np.transpose(np.array(center))
            center = np.matmul(p,center)
            cv2.drawMarker(imgL,(int(center[0]/center[2]),int(center[1]/center[2])),(0,0,255))
    labels.close()

def compute_draw_3D(imgL,label_path,p):
    labels = open(label_path,'r')
    for label in labels:
        field = label.split(' ')
        if field[0] != 'DontCare':
            #Create rotation matrix
            rotation = float(field[14])
            sin = math.sin(rotation)
            cos = math.cos(rotation)
            rot_mat = np.vstack(([cos,0,sin],[0,1,0],[-sin,0,cos]))
            #Create de 3D bbox in real camera coordenates
            l = float(field[10])
            w = float(field[9])
            h = float(field[8])  
            center = field[11:14]
            center = [float(i) for i in center]            
            cube = np.vstack(([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2],[0,0,0,0,-h,-h,-h,-h],[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]))
            offset = np.vstack((np.full((1,8),center[0]),np.full((1,8),center[1]),np.full((1,8),center[2])))
            cube3D = np.matmul(rot_mat,cube) + offset
            cube3D = np.vstack((cube3D,np.full((1,8),1)))
            #Transform to image coordenates and plot
            cube_image = np.matmul(p,cube3D)

            for i in range(8):
                cube_image[0,i] = cube_image[0,i]/cube_image[2,i]
                cube_image[1,i] = cube_image[1,i]/cube_image[2,i]
            
            #Draw points
            #for i in range(8):
            #    cv2.drawMarker(imgL,(int(cube_image[0,i]),int(cube_image[1,i])),(0,0,255))
            #Draw cube

            cv2.line(imgL,(int(cube_image[0,0]),int(cube_image[1,0])),(int(cube_image[0,1]),int(cube_image[1,1])),(0,255,0),1)
            cv2.line(imgL,(int(cube_image[0,0]),int(cube_image[1,0])),(int(cube_image[0,3]),int(cube_image[1,3])),(0,255,0),1)
            cv2.line(imgL,(int(cube_image[0,2]),int(cube_image[1,2])),(int(cube_image[0,1]),int(cube_image[1,1])),(0,255,0),1)
            cv2.line(imgL,(int(cube_image[0,2]),int(cube_image[1,2])),(int(cube_image[0,3]),int(cube_image[1,3])),(0,255,0),1)

            cv2.line(imgL,(int(cube_image[0,4]),int(cube_image[1,4])),(int(cube_image[0,5]),int(cube_image[1,5])),(0,255,0),1)
            cv2.line(imgL,(int(cube_image[0,4]),int(cube_image[1,4])),(int(cube_image[0,7]),int(cube_image[1,7])),(0,255,0),1)
            cv2.line(imgL,(int(cube_image[0,6]),int(cube_image[1,6])),(int(cube_image[0,5]),int(cube_image[1,5])),(0,255,0),1)
            cv2.line(imgL,(int(cube_image[0,6]),int(cube_image[1,6])),(int(cube_image[0,7]),int(cube_image[1,7])),(0,255,0),1)

            cv2.line(imgL,(int(cube_image[0,0]),int(cube_image[1,0])),(int(cube_image[0,4]),int(cube_image[1,5])),(0,255,0),1)
            cv2.line(imgL,(int(cube_image[0,1]),int(cube_image[1,1])),(int(cube_image[0,5]),int(cube_image[1,5])),(0,255,0),1)
            cv2.line(imgL,(int(cube_image[0,2]),int(cube_image[1,2])),(int(cube_image[0,6]),int(cube_image[1,6])),(0,255,0),1)
            cv2.line(imgL,(int(cube_image[0,3]),int(cube_image[1,3])),(int(cube_image[0,7]),int(cube_image[1,7])),(0,255,0),1)


    labels.close()

def bird_view(aereal):

def main():
    path = []

    path.append("/home/miguel/TFG/Stereo-Vision/Datasets/image_left/")
    path.append("/home/miguel/TFG/Stereo-Vision/Datasets/label_2/")
    path.append("/home/miguel/TFG/Stereo-Vision/Datasets/camera_param/calib/")
    path.append("/home/miguel/TFG/Stereo-Vision/Datasets/image_right/")
    i = 10

    image_file = sorted(os.listdir(path[0]))[i]
    label_file = sorted(os.listdir(path[1]))[i]
    param = sorted(os.listdir(path[2]))[i]

    #Read camera params for projections
    p = read_params(path[2]+param)
    #Read Left image
    imgL = cv2.imread(path[0]+image_file,1)
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    assert(imgL is not None)

    #Read labels ,plot 2D bbox and center of objects
    
    #print2D_centers(imgL,path[1]+label_file,p)
    compute_draw_3D(imgL,path[1]+label_file,p)   
     
    
    plt.figure(1)
    plt.imshow(imgL)
    

    imgR = cv2.imread(path[3]+image_file,1)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
    assert(imgR is not None)

    cv2.line(imgR,(int(cube_image[0,0]),int(cube_image[1,0])),(int(cube_image[0,1]),int(cube_image[1,1])),(0,255,0),1)

    plt.figure(2)
    plt.imshow(imgR)
    

    plt.show()
    #stereo = cv2.StereoBM_create(numDisparities=112, blockSize=39)
    #disparity = stereo.compute(imgL,imgR)

    #plt.imshow(disparity,'gray')
    
   





if __name__ == "__main__":
    main()
