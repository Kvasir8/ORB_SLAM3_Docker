import math
import pyrealsense2 as rs
import cv2
import numpy as np
import uuid
import time


from json import dumps
from kafka import KafkaProducer
from statistics import median, mean

im_width=640
im_height=480


def calculate_spatial_distance(object1, object2):

    x1 = (object1.min[0]+object1.max[0])/2
    y1 = (object1.min[1]+object1.max[1])/2
    z1 = object1.min[2]
    a = np.array([x1,y1,z1])

    x2 = (object2.min[0]+object2.max[0])/2
    y2 = (object2.min[1]+object2.max[1])/2
    z2 = object2.min[2]
    b = np.array([x2,y2,z2])

    return np.linalg.norm(a-b)


def get_z_distance(x,y, depth_frame):
    #z_time = time.time()
    z = float(depth_frame[y][x]/1000)
    #z_time = time.time() - z_time

    #print("read depth: " + str(z_time))
    return z

def convert_to_pseudo_thermal(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(15,15),0)
    img_plasma = cv2.applyColorMap(img_blur, cv2.COLORMAP_JET)
    img_neg = cv2.bitwise_not(img_plasma)

    return img_neg

def get_depth_of_center(x, y, width, height, frame):
   
    width = math.ceil(0.5 * width/2)
    height = math.ceil(0.5 * height/2)
    #depths = []
    depths = 0
    cnt = 0

    xmin = x-width
    xmax = x+width
    ymin = y-height
    ymax = y+height

    cropped_frame = frame[xmin:xmax,ymin:ymax]
    return np.mean(cropped_frame)/1000

    #for i in range(1-width,width):    
    #    for j in range(1-height,height):
    #    #    depths.append(get_z_distance(x+i,y+j, frame))
    #        depths += get_z_distance(x+i,y+j, frame)
    #        cnt += 1
#
    #return depths/cnt

    #if len(depths)!=1:
    #    return median(depths)
    #else:
    #    return depths[0]

def generates_msg(data, producer, sender_id): 
    
    topic= "Detections"

    head ={
        "sender_id" : sender_id,
        "sender_type" : 'EAM',
        "msg_id" : str(uuid.uuid4()),
        "timestamp" : int(time.time()),
        "msg_type" : 'notification',
        "msg_content" : 'detections'

    }

    header = []
    for field in head:
        header.append((field,dumps(head[field]).encode('utf-8')))

    msg = []
    for det in data:
        msg_det ={
            "obj_id" : det.obj_id,
            "bbox" :{
                "cs" : "wgs84",
                "min" : det.min,
                "max" : det.max
            },
            "obj_type" : det.obj_type
        }
        msg.append(msg_det)
    try:
    	producer.send(topic, value=msg, headers=header)
    except:
        print("Failed sending message to Kafka")

def get_3D_coordinates(u, v, d, intrinsics):
    
    Fx, Fy, Cx, Cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    
    X = d / Fx * (u - Cx)
    Y = d / Fy * (v - Cy)
    Z = d

    return [X, Y, Z]

def isOnAir():
    return True

def get_average_in_depth(xmin,ymin,xmax,ymax,depth_frame):
    
    width = xmax - xmin
    height = ymax - ymin

    cropped_depth_frame = depth_frame[xmin:xmax, ymin:ymax]
    #shp = cropped_depth_frame.shape
    #m_depth = np.convolve(cropped_depth_frame.ravel(), np.ones((shp[0],shp[1]))/(shp[0]*shp[1]), mode='full')
    #m_depth = np.convolve(cropped_depth_frame.ravel(), np.ones(shp[0]*shp[1])/(shp[0]*shp[1]), mode='full')

    #print(m_depth)

    return np.mean(cropped_depth_frame)/1000
    #depths = []
    #for x in range(xmin,xmin + width):
    #    for y in range(ymin,ymin + height):
    #        if get_z_distance(x,y, depth_frame)!=0:
    #            depths.append(get_z_distance(x,y, depth_frame))
#
    #if depths!=[]:    
    #    return mean(depths)
    #return 0

def get_average_out_depth(xmin,ymin,xmax,ymax,depth_frame):

    width = xmax - xmin
    height = ymax - ymin
    
    depths = []
    for x in range(int(xmin-0.2*width), int(xmin + 1.2*width)):
        for y in range(int(ymin-0.2*height), int(ymin + 1.2*height)):
            if y>=im_height or x>=im_width: continue
            if y<=0 or x<=0: continue
            if y<ymin or y>ymax or x<xmin or x>xmax:
                if get_z_distance(x,y, depth_frame)!=0:
                    depths.append(get_z_distance(x,y, depth_frame))
    if depths!=[]:    
        return median(depths)
    return 5.0



def isVictim(xmin,ymin,xmax,ymax,frame, person, z):

    indepth = get_average_in_depth(xmin,ymin,xmax,ymax,frame)
    #outdepth = get_average_out_depth(xmin,ymin,xmax,ymax,frame)

    diff = indepth - z
    #diff = outdepth -indepth

    #if diff < 0.2:
    if diff < 0.4:
        person.is_victim = True
        person.obj_type = "Victim"
        person.is_responder = False
        return True
    else:
        person.is_responder = True
        person.obj_type = "Responder"
        person.is_victim = False
        return False
