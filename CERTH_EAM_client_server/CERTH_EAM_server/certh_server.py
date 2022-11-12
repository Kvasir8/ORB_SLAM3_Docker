#!/usr/bin/env python
from __future__ import print_function

#http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython


import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import socket
import struct
import numpy as np
import os

from frame_message import Frame
import multiprocessing as mp


class image_converter:

  def __init__(self):   # ros topic publication
    self.bridge = CvBridge()
    self.image_pub = rospy.Publisher("image_topic_2",Image)

    # self.image_sub = rospy.Subscriber("image_topic",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape

    # if cols > 60 and rows > 60 :
    #   cv2.circle(cv_image, (50,50), 10, 255)
    #
    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)



def display_frame(q):
    while True:
        bridge = CvBridge()
        color_pub = rospy.Publisher("/RGBD_color", Image, queue_size = 5)
        depth_pub = rospy.Publisher("/RGBD_depth", Image, queue_size = 5)
        if not q.empty():
            frame = q.get()

            image = frame.image
            depth = frame.depth
            CameraId = str(frame.CameraId.hex())

            # Display RGB and Depth frame (for test purposes)
            colormap_image = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_BONE)
            color_msg = bridge.cv2_to_imgmsg(colormap_image, "bgr8")
            depth_msg = bridge.cv2_to_imgmsg(depth, "mono16")   # CV_16UC1, 16-bit grayscale image  ##http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages
            color_pub.publish(color_msg)
            depth_pub.publish(depth_msg)

            images = np.hstack((image,colormap_image))
            cv2.namedWindow('RGB + Depth from CameraId: ' + CameraId,cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB + Depth from CameraId: ' + CameraId,images)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key==27:
                cv2.destroyAllWindows()
                break

def handle_client(Client, address,q):
    
    # Read messages
    while True:

        data = bytearray()
        new_message = True
        payload = 18

        while True:

            msg = Client.recv(payload-len(data))
            if new_message:

                if msg[:1].hex()!='a5' or msg[1:2].hex()!='5a':
                   continue

                payload = struct.unpack('l',msg[2:10])[0] + 18
                data.extend(msg)
                new_message= False
                continue

            data.extend(msg) 
            if len(data)>=payload:
                break

        # Create frame from messages
        current_frame = Frame(bytes(data))

        # Push to queue
        q.put(current_frame)


#def main():
def main(args):

    # Initialize TCP server

    ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #host = os.environ['IP_REC_FROM'] #'0.0.0.0'
    #port = int(os.environ['PORT_EAM']) #Get port of EAM from .env file
    host ='0.0.0.0'
    port = 4567	# hardcoded



    try:
        ServerSocket.bind((host, port))
    except socket.error as e:
        print(str(e))

    # init ros image with CVbridge
    # ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)

    # Initialize queues and processes
    frame_q = mp.Queue()

    display_process=mp.Process(target=display_frame, args=(frame_q,))
    display_process.start()

    # Listen for connections
    ServerSocket.listen()


    try:
        # rospy.spin()
        while True:
            Client, address = ServerSocket.accept()
            client_process = mp.Process(target=handle_client, args=(Client, address,frame_q))
            client_process.start()
    except KeyboardInterrupt: 
        print("Shutting down")
        client_process.join()
        display_process.join()
        cv2.destroyAllWindows()

if __name__=="__main__":
    #main() 
    main(sys.argv)