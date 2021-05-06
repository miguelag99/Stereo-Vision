import numpy as np

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

    for est in estimations:
        labels_f = open(est.label_id,'r')
        for corners in est.objects:
    
            p1 = corners[0]
            p2 = corners[1]
            p3 = corners[4]
            p4 = corners[5]

            xc = (p1[0]+p2[0]+p3[0]+p4[0])/4
            zc = (p1[2]+p2[2]+p3[2]+p4[2])/4
            center = [xc,zc]
            print("{}{}{}{}".format(p1,p2,p3,p4))

    
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
            
                    print("{}\n GT:{} est:{}".format(cube3D,center_gt,center)) #Si es el mismo objeto (centro cercano) ,se calcula el iou

        labels_f.close()



            
            





