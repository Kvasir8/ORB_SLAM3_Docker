import struct
import numpy as np
import cv2

class Frame:

    def __init__(self, msg):

        self.msg_ts = struct.unpack('d',msg[10:18])[0]
        self.acq_ts = struct.unpack('d',msg[18:26])[0]
        self.rgb_width = struct.unpack('H',msg[26:28])[0]
        self.rgb_height = struct.unpack('H',msg[28:30])[0]
        self.rgb_type = struct.unpack('b',msg[30:31])[0]
        self.rgb_length = struct.unpack('l',msg[31:39])[0]

        self.depth_width = struct.unpack('H',msg[39:41])[0]
        self.depth_height = struct.unpack('H',msg[41:43])[0]
        self.depth_type = struct.unpack('b',msg[43:44])[0]
        self.depth_length = struct.unpack('l',msg[44:52])[0]
        
        self.px  = struct.unpack('d',msg[52:60])[0]
        self.py  = struct.unpack('d',msg[60:68])[0]
        self.pz  = struct.unpack('d',msg[68:76])[0]
        self.qw = struct.unpack('d',msg[76:84])[0]
        self.qx = struct.unpack('d',msg[84:92])[0]
        self.qy = struct.unpack('d',msg[92:100])[0]
        self.qz = struct.unpack('d', msg[100:108])[0]
        self.fov_h = struct.unpack('d',msg[108:116])[0]
        self.fov_v = struct.unpack('d',msg[116:124])[0]
        self.fx = struct.unpack('d',msg[124:132])[0]
        self.fy = struct.unpack('d',msg[132:140])[0]
        self.cx = struct.unpack('d',msg[140:148])[0]
        self.cy = struct.unpack('d',msg[148:156])[0]
        self.rel_alt = struct.unpack('d',msg[156:164])[0]
        self.EquipmentId = struct.unpack('16s', msg[164:180])[0]
        self.CameraId = struct.unpack('16s', msg[180:196])[0]

        self.image = cv2.imdecode(np.fromstring(msg[196:196+self.rgb_length], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        self.depth = np.frombuffer(msg[196+self.rgb_length:196+self.rgb_length+self.depth_length], dtype=np.uint16).reshape((self.depth_height,self.depth_width))
        
        # PNG compression in 3 channels
        #self.depth = cv2.imdecode(np.fromstring(msg[196+self.rgb_length:196+self.rgb_length+self.depth_length], dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
        #B, G, R = np.split(self.depth, [1, 2], 2)
        #self.depth = (B * 256 + G).astype(np.uint16).squeeze()
