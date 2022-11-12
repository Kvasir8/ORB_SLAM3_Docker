import cv2
import socket
import struct
import numpy as np
import os

from frame_message import Frame
import multiprocessing as mp

def display_frame(q):
    while True:
        if not q.empty():
            frame = q.get()

            image = frame.image
            depth = frame.depth
            CameraId = str(frame.CameraId.hex())
            # Display RGB and Depth frame (for test purposes)
            colormap_image = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_BONE)
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


def main():

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

    try:
        while True:
            Client, address = ServerSocket.accept()
            client_process = mp.Process(target=handle_client, args=(Client, address,frame_q))
            client_process.start()
    except KeyboardInterrupt: 
        client_process.join()
        display_process.join()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main() 
