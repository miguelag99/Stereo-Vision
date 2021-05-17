import gi
gi.require_version('Gtk', '2.0')
import argparse
import os
import time

import torch
import numpy as np
import cv2

from PIL import Image
from torch.autograd import Variable
from torchvision.models import vgg


WORK_PATH = os.getcwd()
KITTI_PATH = os.getcwd()+'/Datasets_kitti'
WEIGHTS_PATH = os.getcwd()+'/weights_3D_est'
SAVE_PATH = os.getcwd()+'/results'

from func import *
from plotting import *
from eval import *



def Yolo5_det(imgs):

    torch.hub.set_dir(WORK_PATH)
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    
    
    # Images
    model.classes = [0,1,2,3,5,7] #Person(0),bicycle(1),car(2),motorcycle(3),bus(5),truck(7),traffic light(9),stop(11)
    model.conf = 0.7
    #Adding classes = n in torch.hub.load will change the output layers (must retrain with the new number of classes)

    t_ini = time.time()

    # Inference
    results = model(imgs)
    #print(results.pred)
    print('\n')

    t_end = time.time()
    elapsed = (t_end - t_ini)/len(imgs)
    print("Yolo_time (avg sec per image): {}\n".format(elapsed))

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




def compare_3Dbbox():

    parser = argparse.ArgumentParser()

    parser.add_argument("-ini","--number_init",help ="Num de imagen inicial",type =int)
    parser.add_argument("-end","--number_end",help ="Num de imagen final",type =int)
    parser.add_argument("-eval","--evaluate",help="Evaluar estimacion",action="store_true")
    args = parser.parse_args()
    
    dir1 = KITTI_PATH+'/image_left/'
    dir2 = KITTI_PATH+'/camera_param/calib/'
    dir3 = KITTI_PATH+'/label_2/'

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    names = sorted(os.listdir(dir1))
    par = sorted(os.listdir(dir2))
    gt = sorted(os.listdir(dir3))

    imgs = [dir1 + name for name in names[args.number_init:args.number_end]]     # batch of images  
    
    cal_files = [dir2 + name for name in par[args.number_init:args.number_end]]  # batch of camera params 
    label_files = [dir3 + name for name in gt[args.number_init:args.number_end]] # batch of ground truth 
    
    detections = Yolo5_det(imgs)
    

    imgs = [dir1 + name for name in names[args.number_init:args.number_end]]  # batch of images
    

    #print(imgs) 
    
    #detections.show()
    #detections.save()


    elapsed = 0
    max_t = 0
    min_t = 10

    averages = ClassAverages()
    angle_bins = generate_bins(2)
    model = load_model_est(WEIGHTS_PATH)

    estimations = []

    for i in range(len(imgs)):

        t_ini = time.time()

        truth_img = cv2.imread(imgs[i])
        im = np.copy(truth_img)
        yolo_im = np.copy(truth_img)
        birdview_im = np.zeros((1000,1000,3))

        if(args.evaluate):
            estimations.append(estimation(imgs[i],label_files[i]))
        

        elem_box = detections.pred[i]
        
        for element in elem_box:
        
            name = detections.names[int(element.data[5])]
            #print(name)
            
            if int(element.data[0]) > 5 and int(element.data[2] < (im.shape[1]-5)): #Filter for cropped objects

                calib_file = cal_files[i]
                calib_file = read_params(calib_file) #Params matrix for each image
                

                detectedObject = DetectedObject(im, name,element, calib_file) #New class for each detection to compute theta and crop the detection

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
                dim = dim.cpu().data.numpy()[0, :]
                
                dim += averages.get_item(detected_Class)

                argmax = np.argmax(conf)
                orient = orient[argmax, :]
                cos = orient[0]
                sin = orient[1]
                alpha = np.arctan2(sin, cos)
                alpha += angle_bins[argmax]
                alpha -= np.pi

                location = plot_regressed_3d_bbox(im, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img) #Plot the estimation
                
                #Only objects within z = 40m
                if(location[2]< 40):
                    plot_2d_box(yolo_im,box_2d) #Plot the yolo detection
                    plot_bird_view(birdview_im, dim, alpha, theta_ray,location)

                    if(args.evaluate):

                            orient = alpha + theta_ray
                            R = rotation_matrix(orient)
                            corners = create_corners(dim, location = location, R = R)
                            estimations[i].add_object(corners)

                    #compute_draw_3D(im,label_files[i],proj_matrix) #Draw the kitti 3D bbox.

        t_end = time.time()
        elapsed += (t_end - t_ini)
        if((t_end - t_ini) > max_t):
            max_t = (t_end - t_ini)
        elif((t_end - t_ini) < min_t):
            min_t = (t_end - t_ini)

        
        cv2.putText(birdview_im,"{:.4f} s".format(t_end - t_ini),(50,birdview_im.shape[1]-50),cv2.FONT_HERSHEY_SIMPLEX,1,cv_colors.RED.value,1)
        
        #cv2.imshow("{}".format(i),im)
        #cv2.imshow("{}_birdview".format(i),birdview_im)
        #cv2.imwrite(SAVE_PATH+"/{}.png".format(i),im)
        #cv2.imwrite(SAVE_PATH+"/{}_yolo.png".format(i),yolo_im)
        #cv2.imwrite(SAVE_PATH+"/{}_bird.png".format(i),birdview_im)

    

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    elapsed = elapsed/len(imgs)
    print("Est_time:\nAvg sec per image: {}\nMax:{} Min:{}\n".format(elapsed,max_t,min_t))
    
    if(args.evaluate):
        eval_bird(estimations)


if __name__ == "__main__":
  
    #Yolo5_det()
    compare_3Dbbox()
