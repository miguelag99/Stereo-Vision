import gi
gi.require_version('Gtk', '2.0')
import torch
import os
import numpy as np
import cv2

from PIL import Image
from torch.autograd import Variable
from torchvision.models import vgg


WORK_PATH = os.getcwd()
KITTI_PATH = os.getcwd()+'/Datasets'
WEIGHTS_PATH = os.getcwd()+'/weights_3D_est'
SAVE_PATH = os.getcwd()+'/results'

from func import *
from plotting import *



def Yolo5_det(imgs):

    torch.hub.set_dir(WORK_PATH)
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    
    
    # Images
    model.classes = [0,2,3,5,7] #Person,bicycle(1),car,motorcycle,bus,truck,traffic light(9),stop(11)
    model.conf = 0.5
    #Adding classes = n in torch.hub.load will change the output layers (must retrain with the new number of classes)

    # Inference
    results = model(imgs)
    #print(results.pred)
    print('\n')
    
    return results


def load_model_est(dir):
    model_lst = [x for x in sorted(os.listdir(dir)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        model = Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(dir + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def compare_3Dbbox():
    dir1 = KITTI_PATH+'/image_left/'
    names = sorted(os.listdir(dir1))
    dir2 = KITTI_PATH+'/camera_param/calib/'
    par = sorted(os.listdir(dir2))
    imgs = [dir1 + name for name in names[0:10]]  # batch of images  
    cal_files = [dir2 + name for name in par[0:10]]  # batch of camera params 

    detections = Yolo5_det(imgs)
    imgs = [dir1 + name for name in names[0:10]]  # batch of images
    #print(imgs) 
    
    #detections.show()
    averages = ClassAverages()
    angle_bins = generate_bins(2)
    model = load_model_est(WEIGHTS_PATH)

    for i in range(len(imgs)):
        truth_img = cv2.imread(imgs[i])
        im = np.copy(truth_img)
        
        elem_box = detections.pred[i]
        
        for element in elem_box:
            #print(element)
            name = detections.names[int(element.data[5])]
            #print(name)
            
            calib_file = cal_files[i]
            calib_file = read_params(calib_file)
            detectedObject = DetectedObject(im, name,element, calib_file)

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            if (name == "person"):
                name = "pedestrian"
            
            detected_Class = name
            
            box_2d = [(int(element[0].item()),int(element[1].item())),(int(element[2].item()),int(element[3].item()))]

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            
            #dim += averages.get_item(detected_Class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            location = plot_regressed_3d_bbox(im, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            cv2.imshow("{}".format(i),im)
            cv2.imwrite(SAVE_PATH+"/{}.png".format(i),im)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    #Yolo5_det()
    compare_3Dbbox()