import pandas
import argparse
import os

LABELS = "/home/miguel/TFG/Stereo-Vision/Datasets_kitti/label_2/"
RESULTS = "/home/miguel/TFG/Stereo-Vision/results/"

def kitti_2_csv(label_path):

    files = sorted(os.listdir(label_path))
    col = ['frame','id','type','alpha','left','top','right','bottom','h','w','l','x','y','z','rotation_z','vx','vy','timestamp']
    df = pandas.DataFrame([],columns=col)
    timestamp = 0

    for f in files:

        print(timestamp)

        id = 0

        labels_f = open(LABELS+f,'r')

        for label in labels_f:

            field = label.split(' ')

            if field[0] != 'DontCare':

                if field[0] == "Misc" or field[0] == "Van" or field[0] == "Tram":
                    type = "Other_Vehicle"
                elif field[0] == "Cyclist" :
                    type = "Bike"
                elif field[0] == "Person_sitting":
                    type = "Pedestrian"
                else:
                    type = field[0]

                frame = 0
                
                alpha = field[3]
                (left,top,right,bottom) = (field[4],field[5],field[6],field[7])
                (h,w,l) = (field[8],field[9],field[10])
                (x,y,z) = (field[11],field[12],field[13])
                rotation_z = field[14].split('\n')[0] #Rotation y axis in camera
            
                a = pandas.DataFrame([[frame,id,type,alpha,left,top,right,bottom,h,w,l,x,y,z,rotation_z,0,0,timestamp]],columns=col)
                df = pandas.concat([df,a],axis=0)
                id = id +1
        timestamp = timestamp +1
        labels_f.close()
    
    df.to_csv('kitti_gt.csv',columns=col,index=False)
   




def est_2_csv(est_path):

    files = sorted(os.listdir(est_path))
    col = ['frame','id','type','alpha','left','top','right','bottom','h','w','l','x','y','z','rotation_z','vx','vy','timestamp','score']
    df = pandas.DataFrame([],columns=col)
    timestamp = 0

    for f in files:

        print(timestamp)

        id = 0

        labels_f = open(RESULTS+f,'r')

        for label in labels_f:

            field = label.split(' ')

            if field[0] != 'DontCare':
 
                frame = 0
                if field[0] == "bus":
                    type = "Other_Vehicle"
                else:
                    type = field[0]
             
                alpha = field[3]
                (left,top,right,bottom) = (field[4],field[5],field[6],field[7])
                (h,w,l) = (field[8],field[9],field[10])
                (x,y,z) = (field[11],field[12],field[13])
                rotation_z = field[14] #Rotation y axis in camera
                score = field[15].split('\n')[0]
            
                a = pandas.DataFrame([[frame,id,type,alpha,left,top,right,bottom,h,w,l,x,y,z,rotation_z,0,0,timestamp,score]],columns=col)
                df = pandas.concat([df,a],axis=0)
                id = id +1
        timestamp = timestamp +1
        labels_f.close()
    
    df.to_csv('detections.csv',columns=col,index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-k","--kitti",help="Convert kitti labels to a single csv",action="store_true")
    parser.add_argument("-e","--estimation",help="Convert 3d estimation results to a single csv",action="store_true")
    args = parser.parse_args()

    
    if args.kitti:
        kitti_2_csv(LABELS)

    elif args.estimation:
        est_2_csv(RESULTS)
