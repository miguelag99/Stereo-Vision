import torch
import os

WORK_PATH = '/home/miguel/TFG/Stereo-Vision/Source'
KITTI_PATH = '/home/miguel/TFG/Stereo-Vision/Datasets'

def main():
    torch.hub.set_dir(WORK_PATH)
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Images
    '''
    dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
    imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batch of images
    '''
    dir = KITTI_PATH+'/image_left/'
    names = sorted(os.listdir(dir))
    imgs = [dir + name for name in names[0:50]]  # batch of images    

    model.classes = [0,1,2,3,5,7,9,11] #Person,bibycle,car,motorcycle,bus,truck,traffic light,stop

    #Adding classes = n in torch.hub.load will change the output layers (must retrain with the new number of classes)

    # Inference
    results = model(imgs)
    results.save()  # or .show(), .save()

    


if __name__ == "__main__":
    main()