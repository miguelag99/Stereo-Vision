
import numpy as np
import math

from shapely.geometry import Polygon

from func import *
from plotting import *

class estimation:
    def __init__(self,im_id,label_id):
        self.im_id = im_id
        self.label_id = label_id
        self.objects = []
    def __str__(self):
        return("Image_id:{}\nLabel_id:{}\nObjetos:{}".format(self.im_id,self.label_id,self.objects))
    def add_object(self,obj):
        self.objects.append(obj)



def eval_bird(estimations):

    mean_iou = 0
    mean_d = 0
    i = 0

    for est in estimations:
        labels_f = open(est.label_id,'r')
        print(est.im_id)
        for corners in est.objects:
            p1 = corners[0]
            p2 = corners[1]
            p3 = corners[4]
            p4 = corners[5]
            
            a = (round(np.asscalar(p1[0])*100,2),round(np.asscalar(p1[2])*100,2))
            b = (round(np.asscalar(p2[0])*100,2),round(np.asscalar(p2[2])*100,2))
            c = (round(np.asscalar(p3[0])*100,2),round(np.asscalar(p3[2])*100,2))
            d = (round(np.asscalar(p4[0])*100,2),round(np.asscalar(p4[2])*100,2))

            est_poly = Polygon([a,b,d,c])
           
            xc = (p1[0]+p2[0]+p3[0]+p4[0])/4
            zc = (p1[2]+p2[2]+p3[2]+p4[2])/4
            center = [xc,zc]
    
            for label in labels_f:
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
                    center_gt = field[11:14]
                    center_gt = [float(i) for i in center_gt]            
                    cube = np.vstack(([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2],[0,0,0,0,-h,-h,-h,-h],[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]))
                    offset = np.vstack((np.full((1,8),center_gt[0]),np.full((1,8),center_gt[1]),np.full((1,8),center_gt[2])))
                    cube3D = np.matmul(rot_mat,cube) + offset
                    cube3D = cube3D[0:3,0:4]
        
                    dist = math.sqrt(math.pow(center_gt[0]-center[0],2)+math.pow(center_gt[2]-center[1],2))
                    
                    a = (round(np.asscalar(cube3D[0,0])*100,2),round(np.asscalar(cube3D[2,0])*100,2))
                    b = (round(np.asscalar(cube3D[0,1])*100,2),round(np.asscalar(cube3D[2,1])*100,2))
                    c = (round(np.asscalar(cube3D[0,2])*100,2),round(np.asscalar(cube3D[2,2])*100,2))
                    d = (round(np.asscalar(cube3D[0,3])*100,2),round(np.asscalar(cube3D[2,3])*100,2))
                    gt_poly = Polygon([a,b,c,d])
                       
                    iou = gt_poly.intersection(est_poly).area / gt_poly.union(est_poly).area
                    
                    #Se entiende que si es el mismo objeto el iou debe ser > 0 y los centros deben estar cercanos
                    if(iou>0 and dist < 5):
                        i += 1
                        print("\nIoU{} GT:{} est:{} dist:{}".format(iou,center_gt,center,dist))
                        mean_iou += iou
                        mean_d += dist

        labels_f.close()

    mean_iou *= 1/i
    mean_d *= 1/i
    print("\n\nMean iou:{} Mean dist:{}\n".format(mean_iou,mean_d))



            
            





