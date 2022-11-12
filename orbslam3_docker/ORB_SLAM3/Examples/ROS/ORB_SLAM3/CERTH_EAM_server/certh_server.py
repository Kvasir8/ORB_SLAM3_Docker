#!/usr/bin/env python3
from __future__ import print_function
#http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
import roslib
roslib.load_manifest('ORB_SLAM3')
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

import NotCvBridge as NCB
#import ros_numpy
import std_msgs


class image_converter:

  def __init__(self):   # ros topic publication
    self.bridge = CvBridge()
    self.color_pub = rospy.Publisher("/camera/rgb/image_raw", Image, self.callback, queue_size = 5)
    self.depth_pub = rospy.Publisher("/camera/depth_registered/image_raw", Image, self.callback, queue_size = 5)

  def callback(self,data):
    try:
      cv_image = self.NCB.imgmsg_to_cv2(data, "passthrough")
    except CvBridgeError as e:
      print("CvBridge could not convert images from realsense to opencv")
      print(e)

    (rows,cols,channels) = cv_image.shape

    # if cols > 60 and rows > 60 :
    #   cv2.circle(cv_image, (50,50), 10, 255)
    #
    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)

    try:
      self.color_pub.publish(self.NCB.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)


def display_frame(q):
    bridge = CvBridge()
    #cv_rgb = cv2.imread('Pictures/image.png')
    #cv_depth = cv2.imread('Pictures/image.png', 0)
    color_pub = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=1)
    depth_pub = rospy.Publisher("/camera/depth_registered/image_raw", Image,queue_size=1)

    while True:
        if not q.empty():
            frame = q.get()

            image = frame.image
            depth = frame.depth
            CameraId = str(frame.CameraId.hex())

            # Display RGB and Depth frame (for test purposes)
            colormap_image = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_BONE)
            ## solution for dependency CvBridge issue: https://github.com/ros-perception/vision_opencv/issues/207
            
            #color_msg = np.frombuffer(colormap_image.data, dtype=np.uint8).reshape(image.shape[0], image.shape[1], -1) ## resize it with numpy due to dependency issue
            #color_msg = NCB.cv2_to_imgmsg(colormap_image, "bgr8")
            #color_msg = NCB.cv2_to_imgmsg(cv_rgb, "bgr8")
            color_msg = bridge.cv2_to_imgmsg(colormap_image, "bgr8")
            #color_msg = ros_numpy.msgify(Image, colormap_image)

            depth_msg = bridge.cv2_to_imgmsg(depth, "mono16")   # CV_16UC1, 16-bit grayscale image  ##http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages
            #depth_msg = np.frombuffer(depth.data, dtype=np.uint16).reshape(depth.shape[0], depth.shape[1])
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
    bridge = CvBridge()
    #cv_rgb = cv2.imread('Pictures/image.png')
    #cv_depth = cv2.imread('Pictures/image.png', 0)
    color_pub = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=10)
    depth_pub = rospy.Publisher("/camera/depth_registered/image_raw", Image,queue_size=10)
    
    # Read messages
    while True:

        data = bytearray()
        new_message = True
        payload = 18
        print("Reading msgs")

        while True:
            try:
                msg = Client.recv(payload-len(data))
                if len(msg) == 0:
                    raise socket.error
                if new_message:
                    if msg[:1].hex()!='a5' or msg[1:2].hex()!='5a':
                        print("Check message start")
                        continue
                    payload = struct.unpack('l',msg[2:10])[0] + 18
                    data.extend(msg)    # it will be added for different messages: data getting messages as image array.
                    new_message= False
                    continue
                data.extend(msg) 
                if len(data)>=payload:
                    print("Full message")
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
        current_frame = Frame(bytes(data))  # image data as message 
        print("RGB shape : " + str(current_frame.image.shape))
        print("Depth shape : " + str(current_frame.depth.shape))

        cv2.imshow("Image rgb", current_frame.image)
        cv2.imshow("Image depth", current_frame.depth)
        cv2.waitKey(3)
        #color_msg = bridge.cv2_to_imgmsg(current_frame.image, "bgr8")
        color_msg = NCB.cv2_to_imgmsg(current_frame.image, 'bgr8')
        depth_msg = NCB.cv2_to_imgmsg(current_frame.depth, "mono16")   # CV_16UC1, 16-bit grayscale image  ##http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages
        color_pub.publish(color_msg)
        depth_pub.publish(depth_msg)

        # Push to queue
        q.put(current_frame)


def main():
#def main(args):
    # init ros image with CVbridge
    rospy.init_node('img_conv_realsense', anonymous=True)
    #ic = image_converter()
    #rospy.spin()   ## for subscriber to be used

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
    # Initialize queues and processes
    frame_q = mp.Queue()

    display_process=mp.Process(target=display_frame, args=(frame_q,))
    display_process.start()

    # Listen for connections
    ServerSocket.listen()
    print("Listening connections")

    try:
      while True:
            Client, address = ServerSocket.accept()
            print(address)
            print("Accepted connection")

            client_process = mp.Process(target=handle_client, args=(Client, address,frame_q))
            client_process.start()


    except KeyboardInterrupt: 
        print("Shutting down")
        client_process.join()
        display_process.join()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main() 
    #main(sys.argv)
