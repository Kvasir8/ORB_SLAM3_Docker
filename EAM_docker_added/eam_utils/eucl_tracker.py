import math
from eam_utils.objects import *

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, detections):
        # Objects boxes and ids
        detection_ids = []

        # Get center point of new object
        #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        #print(objects_rect)
        
        for d in detections:
            #print('zzzzzzzzzzzzzzzzzzzzzzzzz')
            #print(rect)
            #x, y, w, h = rect
            x1, y1, x2, y2 = d.bbox[0],d.bbox[1], d.bbox[2], d.bbox[3] 
            cx = int(round((x1+x2)/2))
            cy = int(round((y1+y2)/2))
            diam=math.sqrt((x1-x2)**2+(y1-y2)**2)
            # cx = (int(round(x)) + int(round(x)) + int(round(w))) // 2
            #cy = (int(round(y)) + int(round(y)) + int(round(h))) // 2
            #print('ggggggggggggggggggggggggggggggggg')
            #print(cx)
            #print(cy)

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                #print("yyyyyyyyyyyyyyyyyyyyy")

                #if dist < 25:
                #if dist < 40:
                if dist < 0.25*diam:
                    self.center_points[id] = (cx, cy)
                    #print(self.center_points)
                    #objects_bbs_ids.append([x, y, w, h, id])
                    detection_ids.append(id)

                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                #objects_bbs_ids.append([x, y, w, h, self.id_count])
                detection_ids.append(self.id_count)
                self.id_count += 1
        

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in detection_ids:
            object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        
        
        for i in range(len(detection_ids)):
            detections[i].update_id(detection_ids[i])

        return detections