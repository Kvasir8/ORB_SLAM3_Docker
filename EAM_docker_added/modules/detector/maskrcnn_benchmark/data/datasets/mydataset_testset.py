from maskrcnn_benchmark.structures.bounding_box import BoxList
from PIL import Image , ImageDraw
import torch , time
import pandas as pd
import numpy as np
import ast
from torch.utils.data import Dataset

class MyDataset_testset(Dataset):

    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
        "helicopter",
        "motobike",
        "drone",
    ]

#class MyDataset():
    def __init__(self, transforms=None):
        # as you would do normally
        print("Loading dataset...")

        self.labels = []
        cls = self.CATEGORIES
        self.class_to_idx = dict(zip(cls, range(len(cls))))
        self.idx_to_class = dict(zip(range(len(cls)), cls))
        
        #train_data = pd.read_csv('/home/alouros/vscode/Drone_FasterRCNN/thermal_train_dataset.csv')
        #train_data['Labels'] = train_data['Labels'].apply(lambda x: ast.literal_eval(x))
        #train_data['Boxes'] = train_data['Boxes'].apply(lambda x: ast.literal_eval(x))

        test_data = pd.read_csv('/home/alouros/vscode/Drone_FasterRCNN/thermal_test_dataset_temp.csv')
        test_data['Labels'] = test_data['Labels'].apply(lambda x: ast.literal_eval(x))
        test_data['Boxes'] = test_data['Boxes'].apply(lambda x: ast.literal_eval(x))
        
        #val_data = pd.read_csv('thermal_val_dataset.csv')
        #val_data['Labels'] = val_data['Labels'].apply(lambda x: ast.literal_eval(x))
        #val_data['Boxes'] = val_data['Boxes'].apply(lambda x: ast.literal_eval(x))

        self.paths = test_data['Path'].tolist()
        self.photos = test_data['JPG'].tolist()
        for i in range(len(self.photos)):
            self.photos[i] = '/home/alouros/vscode/LSOTB-TIR_intrested_thermal_colorjet/LSOTB-TIR_TrainingData/TrainingData' + self.paths[i] + self.photos[i]

        self.img_width = test_data['Width'].tolist()
        self.img_height = test_data['Height'].tolist()
        self.objects = test_data['Num of Objects'].tolist()

        lbls = test_data['Labels'].tolist()
        labs = []
        for lbl in lbls:
            labs = []
            for l in lbl:
                labs.append(self.class_to_idx[l])
            self.labels.append(labs)

        self.bboxes = test_data['Boxes'].tolist()
        self.id_to_img_map = list(range(test_data.size))
        self.transforms = transforms


    def __getitem__(self, idx):
        # load the image as a PIL Image

        image =Image.open(self.photos[idx].strip())
        #Just for test
        #if idx==50010:
        #    shape =[(self.bboxes[idx][0][0],self.bboxes[idx][0][1]),(self.bboxes[idx][0][2],self.bboxes[idx][0][3])]
        #    image1 = ImageDraw.Draw(image)
        #    image1.rectangle(shape, outline ="red")
        #    image.show() 

        
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes = self.bboxes[idx]
        
        # and labels
        labels = torch.tensor(self.labels[idx])
        
        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        #if self.transforms:
        #    image, boxlist = self.transforms(image, boxlist)
        if self.transforms is not None:
            image, boxlist = self.transforms(image, boxlist)
        
        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def __len__(self):
        return len(self.photos)

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": self.img_height[idx], "width": self.img_width[idx]}


    def get_groundtruth(self, idx):
        # load the image as a PIL Image
        image =Image.open(self.photos[idx])

        boxes = self.bboxes[idx]
        
        # and labels
        labels = torch.tensor(self.labels[idx])
        
        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        #if self.transforms:
        #    image, boxlist = self.transforms(image, boxlist)

        if self.transforms is not None:
            image, boxlist = self.transforms(image, boxlist)

        return boxlist


    #def get_img_name(self,idx):
    #    return self.photos[idx]

#just for test
#start_time = time.time()
#test = MyDataset()
##
#for i in range(235235):
#    print(test.__getitem__(i))
#    print(test.get_img_info(i))
#    print(test.get_img_name(i))
#    print(i)
#    #test.__getitem__(i)
#
#print("----{} seconds ----".format(time.time()-start_time))

