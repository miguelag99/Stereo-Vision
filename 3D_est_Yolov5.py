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



def Yolo5_det(imgs,model):

    # Images
    model.classes = [0,1,2,3,5,7] #Person(0),bicycle(1),car(2),motorcycle(3),bus(5),truck(7),traffic light(9),stop(11)
    model.conf = 0.6
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




def execute():

    conf_threshold = 0.45

    dir1 = KITTI_PATH+'/image_left/'
    dir2 = KITTI_PATH+'/camera_param/calib/'
    dir3 = KITTI_PATH+'/label_2/'

 
    names = sorted(os.listdir(dir1))
    #print(names[0:len(names)-1])
    par = sorted(os.listdir(dir2))
    gt = sorted(os.listdir(dir3))

    imgs = [dir1 + name for name in names[args.number_init:args.number_end]]     # batch of images  
    
    cal_files = [dir2 + name for name in par[args.number_init:args.number_end]]  # batch of camera params 
    label_files = [dir3 + name for name in gt[args.number_init:args.number_end]] # batch of ground truth 
    
    torch.hub.set_dir(WORK_PATH)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    detections = Yolo5_det(imgs,yolo_model)
    

    imgs = [dir1 + name for name in names[args.number_init:args.number_end]]  # batch of images
      
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
        gt_im = np.copy(truth_img)
        im = np.copy(truth_img)
        yolo_im = np.copy(truth_img)
        birdview_im = np.zeros((1000,1000,3))

        #print("\n\nImage:{}\n".format(imgs[i]))

        elem_box = detections.pred[i]
        
        for element in elem_box:
        
            name = detections.names[int(element.data[5])]
            yolo_conf = element[4]

            
            if int(element.data[0]) > 5 and int(element.data[2] < (im.shape[1]-5)): #Filter for cropped objects

                calib_file = cal_files[i]
                calib_file = read_params(calib_file) #Params matrix for each image

                # calib_file = np.vstack(([671.5123291015625, 0.0, 664.2958374023438, 0.0],[0.0, 671.5123291015625, 347.06634521484375, 0.0],[0.0, 0.0, 1.0, 0.0]))
                
                #print(calib_file)
                #print(element)
		
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
                conf = conf[argmax]
                orient = orient[argmax, :]
                cos = orient[0]
                sin = orient[1]
                alpha = np.arctan2(sin, cos)
                alpha += angle_bins[argmax]
                alpha -= np.pi

                location = plot_regressed_3d_bbox(im, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img) #Plot the estimation
                
                # print("Loc:{}".format(location))
                # print("Theta_ray:{} Orient:{} Alpha:{}".format(theta_ray,orient,alpha))

                # Alpha es el ángulo que forma la unión del centro del objeto a la cámara con el eje x 

                #Calcular la score ponderando con la distancia 
                
                if conf > 0.8:
                    
                    conf = pow((1.5*conf-0.5),3)

                file_name = SAVE_PATH+"/"+label_files[i].split("/")[7]
                # file_name = SAVE_PATH+"/"+"{:010d}.txt".format(i)
                detection_2_file(file_name,name,theta_ray,element.data,dim,location,alpha,conf)

                plot_2d_box(yolo_im,box_2d) #Plot the yolo detection
                #compute_draw_3D(gt_im,label_files[i],proj_matrix) #Draw the kitti 3D bbox alone.
               
                if(conf> conf_threshold):
                    #compute_draw_3D(im,label_files[i],proj_matrix) #Draw the kitti 3D bbox within the same image.        
                    plot_bird_view(birdview_im, dim, alpha, theta_ray,location)

                

            # else:
            #     print("Objeto fuera de rango")

        
        
        t_end = time.time()
        elapsed += (t_end - t_ini)
        if((t_end - t_ini) > max_t):
            max_t = (t_end - t_ini)
        elif((t_end - t_ini) < min_t):
            min_t = (t_end - t_ini)

        
        cv2.putText(birdview_im,"{:.4f} s".format(t_end - t_ini),(50,birdview_im.shape[1]-50),cv2.FONT_HERSHEY_SIMPLEX,1,cv_colors.RED.value,1)
        
        #cv2.imshow("{}".format(i),im)
        #cv2.imshow("{}_birdview".format(i),birdview_im)
        #cv2.imshow("{}_kitti_gt".format(i),gt_im)
        #cv2.imwrite(SAVE_PATH+"/{}_gt.png".format(i),gt_im)
        cv2.imwrite(SAVE_PATH+"/{}.png".format(i),im)
        cv2.imwrite(SAVE_PATH+"/{}_yolo.png".format(i),yolo_im)
        #cv2.imwrite(SAVE_PATH+"/{}_bird.png".format(i),birdview_im)

    

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    elapsed = elapsed/len(imgs)
    print("Est_time:\nAvg sec per image: {}\nMax:{} Min:{}\n".format(elapsed,max_t,min_t))

 
    
    
        


def eval_Kitti():

    im_per_batch = 50
    kitti_iou = 0
    kitti_d = 0
    conf_threshold = 0.45
    max_time = 0

    dir1 = KITTI_PATH+'/image_left/'
    dir2 = KITTI_PATH+'/camera_param/calib/'
    dir3 = KITTI_PATH+'/label_2/'

    names = sorted(os.listdir(dir1))
    par = sorted(os.listdir(dir2))
    gt = sorted(os.listdir(dir3))

    n_batch = np.floor(len(names)/im_per_batch)

    torch.hub.set_dir(WORK_PATH)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

    averages = ClassAverages()
    angle_bins = generate_bins(2)
    model = load_model_est(WEIGHTS_PATH)

    for b in range(n_batch.astype(int)):

        print("\nBatch n {} / {}".format(b,n_batch.astype(int)))

        imgs = [dir1 + name for name in names[b*im_per_batch:(b+1)*im_per_batch]]     # batch of images  
        cal_files = [dir2 + name for name in par[b*im_per_batch:(b+1)*im_per_batch]]  # batch of camera params 
        label_files = [dir3 + name for name in gt[b*im_per_batch:(b+1)*im_per_batch]] # batch of ground truth 

        detections = Yolo5_det(imgs,yolo_model)

        imgs = [dir1 + name for name in names[b*im_per_batch:(b+1)*im_per_batch]]     # batch of images  

        estimations = []
    
        for i in range(len(imgs)):

            t_ini = time.time()
            truth_img = cv2.imread(imgs[i])
            im = np.copy(truth_img)

            estimations.append(estimation(imgs[i],label_files[i]))
    
            elem_box = detections.pred[i]
            
            for element in elem_box:
            
                name = detections.names[int(element.data[5])]
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
                    
                    conf = conf[argmax]
                    conf = conf/(1+(0.025*location[2]))

                    file_name = SAVE_PATH+"/"+label_files[i].split("/")[7]
                    detection_2_file(file_name,name,theta_ray,element.data,dim,location,alpha,conf)

                    if(conf> conf_threshold):
  
                        orient = alpha + theta_ray
                        R = rotation_matrix(orient)
                        corners = create_corners(dim, location = location, R = R)
                        estimations[i].add_object(corners)

            t_end = time.time()

            if((t_end - t_ini) > max_time):
                max_time = (t_end - t_ini)


        
        (batch_iou,batch_d) = eval_bird(estimations)

        if(kitti_iou == 0):
            kitti_iou = batch_iou
            kitti_d = batch_d           
        else:
            kitti_iou = (batch_iou+kitti_iou)/2
            kitti_d = (batch_d+kitti_d)/2
        print("Accumulated mean Iou: {} and accumulated mean dist: {}".format(kitti_iou,kitti_d))

    print("Se han analizado {} imagenes con una conf min de {}, obteniendo Mean IoU: {} , Mean dist: {} y t_max de ejecución del estimador de {}".format(n_batch*im_per_batch,conf_threshold, kitti_iou, kitti_d,max_time))



if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("-ini","--number_init",help ="Num de imagen inicial",type =int)
    parser.add_argument("-end","--number_end",help ="Num de imagen final",type =int)
    parser.add_argument("-eval","--evaluate",help="Evaluar estimacion",action="store_true")
    args = parser.parse_args()
    
    os.system("rm -rf "+SAVE_PATH)
    os.makedirs(SAVE_PATH)
    

    if(args.evaluate):
        create_save_files(source_path="/home/miguel/TFG/Stereo-Vision/Datasets_kitti/label_2",dest_path="/home/miguel/TFG/Stereo-Vision/results")
        print("Evaluación sobre Kitti")
        eval_Kitti()
    else:
        execute()
