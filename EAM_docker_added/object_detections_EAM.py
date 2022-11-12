from logging import exception
import cv2
import socket
import struct
import threading
import json 
import time
import os
import numpy as np
import multiprocessing as mp
import uuid
from dotenv import load_dotenv

from eam_utils.eucl_tracker import EuclideanDistTracker
#from modules.detector.predictor import COCODemo
from kafka import KafkaProducer
from eam_utils.objects import Person, Object
from eam_utils.functions import*
from eam_utils.frame_message import Frame
import torch

from queue import Queue,Empty

#from Queue import Empty
class ClearableQueue(Queue):

    def clear(self):
        try:
            while True:
                self.get_nowait()
        except Empty:
            pass


def display_frame(det_q):
    while True:
        if not det_q.empty():
            frame=det_q.get()

            image=frame[0]
            depth=frame[1]
            CameraId= str(frame[2].hex())
            # Display RGB and Depth frame (for test purposes)
            colormap_image=cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_BONE)
            images=np.hstack((image,colormap_image))
            cv2.namedWindow('RGB + Depth '+CameraId,cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB + Depth '+CameraId,images)
            key= cv2.waitKey(1)
            if key & 0xFF == ord('q') or key==27:
                cv2.destroyAllWindows()
                break

def handle_client(s,q):

    ClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = True

    try:        
        ClientSocket.connect(s)
        print("Connected to " + str(s[0]))
    except socket.error:
        print("Could not connect to " + str(s[0]))
        connected = False
        ClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while not connected:
            try:
                ClientSocket.connect(s)
                connected=True
                print("Connection established")
            except socket.error:
                time.sleep(2)
                print("Trying to connect to " + str(s[0]))
    

    # Read messages
    while True:

        data = bytearray()
        new_message = True
        payload = 18
        #print("Reading")

        while True:
            try:
#                print(payload)
#                print(len(data))
                msg = ClientSocket.recv(payload-len(data))
                if len(msg) == 0:
                    raise socket.error

                if new_message:

                    if msg[:1].hex()!='a5' or msg[1:2].hex()!='5a':
                       #print("Check message start...")
                       continue
                    payload = struct.unpack('l',msg[2:10])[0] + 18
                    data.extend(msg)
                    new_message= False
                    continue

                data.extend(msg)

                if len(data)>=payload:
                    #print("Full message")
                    break
            except socket.error as e:
                print(str(e))
                print("Connection lost with " + str(s[0]))
                connected=False
                new_message = True
                ClientSocket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                while not connected:
                    try:
                        ClientSocket.connect(s)
                        #ClientSocket.setblocking(0)
                        connected=True
                        print("Reconnected to " + str(s[0]))
                    except socket.error:
                        time.sleep(5)
                        print("Trying to reconnect to " + str(s[0]))

        # Create frame from messages
        current_frame = Frame(bytes(data))
        #print("RGB shape : " + str(current_frame.image.shape))
        #print("Depth shape : " + str(current_frame.depth.shape))

        # Push to queue
        q.put(current_frame)
        print("Frame pushed to queue")
        
def run_detections(frame_q,det_q):

    #Initializations
    #detector = COCODemo(min_image_size=640, confidence_threshold=0.7)
    detector = torch.hub.load(
        'modules/detector/yolov5', 
        'custom', path='modules/detector/yolo_best.pt', source='local'
    )
    #producer = KafkaProducer(bootstrap_servers=[str(os.environ['IP_KAFKA']) + ':' + str(os.environ['PORT_KAFKA'])],
    #                         value_serializer=lambda x:
    #                        json.dumps(x).encode('utf-8'))
    sender_id = str(uuid.uuid4())

    start_time = time.time()
    unique_dets=[]
    counter_uniq_object_id=0
    last_uniq_object_id=0

    frame_cnt = 0

    while True:

        if not frame_q.empty():
            
            frame = frame_q.get()
            #print(frame.depth.shape)
            
            start_inf_time = time.time()
            #result, result_labels, result_scores, result_bboxes = detector.run_on_opencv_image(frame.image)
            results = detector(frame.image)
            results_list = results.pandas().xyxy[0]
            result_labels = results_list["name"].values.tolist()
            result_scores= results_list["confidence"].values.tolist()
            result_bboxes = results_list[["xmin","ymin","xmax","ymax"]].values.tolist()
            end_inf_time = time.time() - start_inf_time

#            print("Inf time : " + str(end_inf_time))

            detections = []
            if result_labels!=[]:
                for x in range(len(result_labels)):

                    # Calculate depth of area around object center
                    xc = int(round(result_bboxes[x][0]+result_bboxes[x][2])//2)
                    yc = int(round(result_bboxes[x][1]+result_bboxes[x][3])//2)
                    object_width = round(result_bboxes[x][2]-result_bboxes[x][0])
                    object_height = round(result_bboxes[x][3]-result_bboxes[x][1])
                    z = get_depth_of_center(xc,yc,object_width, object_height, frame.depth)

                    z_time = time.time() - start_inf_time - end_inf_time
 #                   print("Z time: " + str(z_time))
                    #z = 0.3
                    if z==0: # do not calculate object to close on camera
                        continue

                    ## Create Detection object
                    #obj = Person(result_labels[x],result_scores[x],result_bboxes[x],x,z,[frame.fx,frame.fy,frame.cx,frame.cy],[frame.px,frame.py,frame.pz], [frame.qx,frame.qy,frame.qz, frame.qw])
                    #isVictim(int(result_bboxes[x][0]),int(result_bboxes[x][1]),int(result_bboxes[x][2]),int(result_bboxes[x][3]), frame.depth, obj, z)
                    #victim_time = time.time() - start_inf_time - end_inf_time - z_time
                    #print("victim time: " + str(victim_time))
                    #if result_labels[x]=="person":
                    #    detections.append(obj)

                    # Create Detection object
                    #obj = Person(result_labels[x],result_scores[x],result_bboxes[x],x,z,[frame.fx,frame.fy,frame.cx,frame.cy],[frame.px,frame.py,frame.pz], [frame.qx,frame.qy,frame.qz, frame.qw])
                    

                    if result_labels[x]=="person":
                        obj = Person(result_labels[x],result_scores[x],result_bboxes[x],x,z,[frame.fx,frame.fy,frame.cx,frame.cy],[frame.px,frame.py,frame.pz], [frame.qx,frame.qy,frame.qz, frame.qw])
                        
                        isVictim(int(result_bboxes[x][0]),int(result_bboxes[x][1]),int(result_bboxes[x][2]),int(result_bboxes[x][3]), frame.depth, obj, z)
                    else:
                        obj = Object(result_labels[x],result_scores[x],result_bboxes[x],x,z,[frame.fx,frame.fy,frame.cx,frame.cy],[frame.px,frame.py,frame.pz], [frame.qx,frame.qy,frame.qz, frame.qw])

                    victim_time = time.time() - start_inf_time - end_inf_time - z_time
  #                  print("victim time: " + str(victim_time))
                    detections.append(obj) 


                    

            #det_time = time.time() - start_inf_time - end_inf_time - z_time - victim_time
            #print("Calculations time: " +str(det_time))

            if detections!=[]:
                if unique_dets!=[]:
                    for det in detections:
                        det.draw_detection(frame.image)
                        exists = False
                        for uniq in unique_dets:
                            if calculate_spatial_distance(det,uniq) < 0.2:
                                exists = True
                                break
                        if not exists:
                            det.update_id(counter_uniq_object_id)
                            counter_uniq_object_id +=1
                            unique_dets.append(det)
                else:
                    for det in detections:
                        det.draw_detection(frame.image)
                        det.update_id(counter_uniq_object_id)
                        counter_uniq_object_id +=1
                        unique_dets.append(det)

            #uni_time = time.time() - start_inf_time - end_inf_time - z_time - victim_time
            #print("Finding uniques time: " + str(uni_time))
            frame_cnt += 1
            if frame_cnt < 10:
                while not frame_q.empty():
                    frame_q.get_nowait()
                continue
            
            det_q.put([frame.image,frame.depth,frame.CameraId])
            print("Detection pushed to queue")

        if (time.time() - start_time) > 9:
            #print("10 seconds passed")
            
            if unique_dets!=[]:
                # Send new detections over Kafka
                if last_uniq_object_id==0:
                    #kafka_thread = threading.Thread(name='non-daemon', target=generates_msg(unique_dets,producer,sender_id))
                    #kafka_thread.start()
                    #print("Sent detections to Kafka(1st)")
                    last_uniq_object_id= counter_uniq_object_id
                else:
                    if unique_dets[last_uniq_object_id:]!=[]:
                        #kafka_thread = threading.Thread(name='non-daemon', target=generates_msg(unique_dets[last_uniq_object_id:],producer,sender_id))
                        #kafka_thread.start()
                        #print("Sent detections to Kafka")
                        last_uniq_object_id= counter_uniq_object_id
            
            #unique_dets = [] #Set list of detection object as empty eath time send object to kafka
            start_time = time.time()



def main():

    #load_dotenv()

    # Initialize queues and processes
    display = True
    frame_q = mp.Queue()
    #frame_q = ClearableQueue()
    det_q = mp.Queue()
    #det_q = ClearableQueue()

    detector_process = mp.Process(target=run_detections, args=(frame_q, det_q))
    detector_process.start()
    if display:
        display_process=mp.Process(target=display_frame, args=(det_q,))
        display_process.start()
    
    # Initialize TCP server

    socks = []
    procs = []
    
    host_list = os.getenv('SERVER_IPS').split(',')
    port_list = os.getenv('SERVER_PORTS').split(',')
    for i in range(len(host_list)):
        socks.append((host_list[i],int(port_list[i])))
    
    for i in range(len(socks)):
        procs.append(mp.Process(target=handle_client, args=(socks[i], frame_q)))
        procs[i].start()

    for p in procs:
        p.join()
    detector_process.join()
    if display:
        display_process.join()
    cv2.destroyAllWindows()

if __name__=="__main__":

    main() 
