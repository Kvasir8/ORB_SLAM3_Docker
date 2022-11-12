from __future__ import division
import cv2
import numpy as np
import socket
import struct
from numpy.lib.function_base import append
import pyrealsense2 as rs
import time
import uuid
import os

def get_rs_frames(pipeline, align):
        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                frame_timestamp= color_frame.get_timestamp()

                if not aligned_depth_frame or not color_frame:
                    continue
                    
                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_BONE)

                return color_image ,depth_image, frame_timestamp/1000

        except RuntimeError as err:
            print(err)
            pass

def main():

    # Initialize Realsense
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        ctx = rs.context()
        serial=ctx.devices[0].get_info(rs.camera_info.serial_number)

        # Equipment and Camera GUID (random for test purposes)
        EquipmentId=  uuid.uuid4()
        CameraId = uuid.uuid4()

        # Initialize RGB and Depth Streams # needs to be finxed via rosbag img resolution

        width = 640
        height = 480
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 6)

        profile = pipeline.start(config)
        prof = profile.get_stream(rs.stream.color)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        #depth_sensor = profile.get_device().first_depth_sensor()
        #depth_scale = depth_sensor.get_depth_scale()

        # Align RGB and Depth streams
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Initialize TCP socket

        ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #host= str(os.environ['IP_EAM']) #Get IP of EAM from .env file 
        #port = int(os.environ['PORT_EAM']) #Get port of EAM from .env file
        host = '127.0.0.1'
        port = 23098

        try:
            ServerSocket.bind((host, port))
        except socket.error as e:
            print(str(e))


        ServerSocket.listen()
        print("Listening connections")
        Server, address = ServerSocket.accept()
        print(address)
        print("Accepted connection")
        cnt = -1

        while True:
            # Get RGB and Depth frames from Realsense

            rgb_frame, depth_frame, frame_ts = get_rs_frames(pipeline, align)
            cnt += 1
            if cnt>3:
                cnt = -1
                # Compress RGB 
                rgb_png = cv2.imencode('.png', rgb_frame)[1]
                rgb_length = len(rgb_png)
                rgb_encode = np.array(rgb_png)
                rgb_msg = rgb_encode.tobytes()
                # Compress Depth 
                #depth_compress = np.zeros((height,width, 3), np.int)
                #depth_compress[:,:,0]= depth_frame // 256
                #depth_compress[:,:,1]= depth_frame % 256
                #depth_png = cv2.imencode('.png', depth_compress)[1]
                #depth_length = len(depth_png)
                #depth_encode = np.array(depth_png)
                #depth_msg = depth_encode.tobytes()
                depth_msg = depth_frame.tobytes()
                depth_length = len(depth_msg)
                print(depth_length)
                # Camera intrinsics
                intrin = prof.as_video_stream_profile().get_intrinsics()
                # Camera extrinsics
                px  = float(5.5)
                py  = float(8.5)
                pz  = float(3.5)
                qw = float(3.00)
                qx = float(90.2)
                qy = float(54.3)
                qz = float(4.34)
                fov_h = float(66)
                fov_v = float(40)
                rel_alt = float(200)
                # Connect to server
                # Pack message
                payload = 178 + rgb_length + depth_length
                msg = bytearray()
                msg.extend(bytes.fromhex('a5'))
                msg.extend(bytes.fromhex('5a'))
                msg.extend(struct.pack('l',payload))
                msg.extend(struct.pack('d',time.time()))
                msg.extend(struct.pack('d',frame_ts))
                msg.extend(struct.pack('H',width))
                msg.extend(struct.pack('H',height))
                msg.extend(struct.pack('b',3))
                msg.extend(struct.pack('l',rgb_length))
                msg.extend(struct.pack('H',width))
                msg.extend(struct.pack('H',height))
                msg.extend(struct.pack('b',2))
                msg.extend(struct.pack('l',depth_length))
                msg.extend(struct.pack('d',px))
                msg.extend(struct.pack('d',py))
                msg.extend(struct.pack('d',pz))
                msg.extend(struct.pack('d',qw))
                msg.extend(struct.pack('d',qx))
                msg.extend(struct.pack('d',qy))
                msg.extend(struct.pack('d',qz))
                msg.extend(struct.pack('d',fov_h))
                msg.extend(struct.pack('d',fov_v))
                msg.extend(struct.pack('d',intrin.fx))
                msg.extend(struct.pack('d',intrin.fy))
                msg.extend(struct.pack('d',intrin.ppx))
                msg.extend(struct.pack('d',intrin.ppy))
                msg.extend(struct.pack('d',rel_alt))
                msg.extend(struct.pack('16s',EquipmentId.bytes))
                msg.extend(struct.pack('16s',CameraId.bytes))
                msg.extend(rgb_msg)
                msg.extend(depth_msg)
                # Send message
                Server.send(bytes(msg))
                print("Frame sent")
    except KeyboardInterrupt:
        print("Closing server")
        ServerSocket.close()
        time.sleep(5)
        print("Server closed")

if __name__ == "__main__":
    main()

