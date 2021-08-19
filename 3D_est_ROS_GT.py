import gi
from torch.functional import split
gi.require_version('Gtk', '2.0')
import argparse
import os
import time

import torch
import numpy as np
import cv2
import pandas

from PIL import Image
from torch.autograd import Variable
from torchvision.models import vgg


WORK_PATH = os.getcwd()
# GT_PATH = os.getcwd()+'/gt_rosbag/camera/'
GT_PATH = "/home/miguel/shared_home/perception/camera/"
WEIGHTS_PATH = os.getcwd()+'/weights_3D_est'
# SAVE_PATH = os.getcwd()+'/results_ROS'



from func import *
from plotting import *
from eval import *



def Yolo5_det(imgs,model):

    # Images
    model.classes = [0,1,2,3,5,7] #Person(0),bicycle(1),car(2),motorcycle(3),bus(5),truck(7),traffic light(9),stop(11)
    model.conf = 0.1
    #Adding classes = n in torch.hub.load will change the output layers (must retrain with the new number of classes)

    t_ini = time.time()

    # Inference
    results = model(imgs)
    #print(results.pred)
    #print('\n')

    t_end = time.time()
    elapsed = (t_end - t_ini)/len(imgs)
    print("Yolo_time (avg sec per image): {}".format(elapsed))

    return results


def load_model_est(dir):


    model_lst = [x for x in sorted(os.listdir(dir)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file
        model = Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(dir + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model


def detect_ROS():

    im_per_batch = 15
    conf_threshold = 0.4
  
    # Camera offset
    h_off = 1.57/2
    l_off = 0.41

    dir1 = GT_PATH + '/data/'
    dir2 = GT_PATH + '/intrinsic_matrix.txt'

    names = sorted(os.listdir(dir1))
    
    intrinsic = load_P_mat(dir2)
    #print(intrinsic)

    timestamps = load_timestamps(GT_PATH+'timestamp.txt')
    
    n_batch = np.ceil(len(names)/im_per_batch)

    torch.hub.set_dir(WORK_PATH)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

    averages = ClassAverages()
    angle_bins = generate_bins(2)
    model = load_model_est(WEIGHTS_PATH)

    col = ['frame','timestamp','id','type','alpha','left','top','right','bottom','h','w','l','x','y','z','rotation_z','vx','vy','score']
    df = pandas.DataFrame([],columns=col)

    for b in range(n_batch.astype(int)):

        print("\nBatch n {} / {}".format(b,n_batch.astype(int)))
        print(b)

        if b != (n_batch.astype(int)-1):
            imgs = [dir1 + name for name in names[b*im_per_batch:(b+1)*im_per_batch]]     # batch of images  
            detections = Yolo5_det(imgs,yolo_model)
            imgs = [dir1 + name for name in names[b*im_per_batch:(b+1)*im_per_batch]]     # batch of images  
            times = timestamps[b*im_per_batch:(b+1)*im_per_batch]
        else:
            imgs = [dir1 + name for name in names[b*im_per_batch:]]     # batch of images  
            detections = Yolo5_det(imgs,yolo_model)
            imgs = [dir1 + name for name in names[b*im_per_batch:]]     # batch of images 
            times = timestamps[b*im_per_batch:] 

        
        for i in range(len(imgs)):

            t = times[i]
            id = 0
            truth_img = cv2.imread(imgs[i])
            im = np.copy(truth_img)
    
            elem_box = detections.pred[i]
            
            for element in elem_box:
            
                name = detections.names[int(element.data[5])]
                if int(element.data[0]) > 5 and int(element.data[2] < (im.shape[1]-5)): #Filter for cropped objects

                    detectedObject = DetectedObject(im, name,element, intrinsic) #New class for each detection to compute theta and crop the detection

                    theta_ray = detectedObject.theta_ray
                    input_img = detectedObject.img
                    proj_matrix = detectedObject.proj_matrix
                        
                    detected_Class = name
                    
                    box_2d = [(int(element[0].item()),int(element[1].item())),(int(element[2].item()),int(element[3].item()))] #2D Bbox as a topule of points

                    input_tensor = torch.zeros([1,3,224,224]).cuda()
                    input_tensor[0,:,:,:] = input_img

                    [orient, conf, dim] = model(input_tensor) #Apply the model to get the estimation

                    orient = orient.cpu().data.numpy()[0, :, :]
                    conf = conf.cpu().data.numpy()[0, :]
                    conf = max(conf)*0.8
                    dim = dim.cpu().data.numpy()[0, :]
                    
                    dim += averages.get_item(detected_Class)

                    argmax = np.argmax(conf)
                    orient = orient[argmax, :]
                    cos = orient[0]
                    sin = orient[1]
                    alpha = np.arctan2(sin, cos)
                    alpha += angle_bins[argmax]
                    alpha -= np.pi

                    if name != "Pedestrian":
                        name = "Car"
                    
                    if name == "car":
                        name = "Car"


                    location = plot_regressed_3d_bbox(im, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img) #Plot the estimation

                    # print("Loc:{}".format(location))
                    
                    # conf = conf[argmax]
                    # conf = conf/(1+(0.1*location[2]))

                    # Print 2D and 3D estimations of a det batch

                    # if (b==0):

                    #     plot_2d_box(im,box_2d) #Plot the yolo detection
                    #     cv2.imshow("{}".format(i),im)

                    # if(conf> conf_threshold):
                    if(conf> 0):
                     
                        orient = (alpha + theta_ray).cpu()
                        a = pandas.DataFrame([[((b*im_per_batch)+i),t,id,name,alpha,box_2d[0][0],box_2d[0][1],box_2d[1][0],box_2d[1][1]\
                            ,dim[0],dim[1],dim[2],location[2],-location[0],-location[1],orient.item(),0,0,conf]],columns=col)
                        df = pandas.concat([df,a],axis=0)
                        id = id + 1

    df.to_csv('ROS_det.csv',columns=col,index=False)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

       

def load_P_mat(file_path):

    camera_param = open(file_path,'r')
    matrix = camera_param.readlines()
    camera_param.close()
    matrix = matrix[0].split(',') 
    matrix = [float(i) for i in matrix]
    
    p = np.vstack((matrix[0:4],matrix[4:8],matrix[8:12]))
    return p


def load_timestamps(file_path):

    times = open(file_path,'r')
    values = times.readlines()
    values = [float(i.split('\n')[0]) for i in values]
    return values



if __name__ == "__main__":

    detect_ROS()
    
    
