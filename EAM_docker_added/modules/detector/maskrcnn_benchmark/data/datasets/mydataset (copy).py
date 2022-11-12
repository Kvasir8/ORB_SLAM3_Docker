from maskrcnn_benchmark.structures.bounding_box import BoxList
from PIL import Image , ImageDraw
import torch , time
from torch.utils.data import Dataset

class MyDataset(Dataset):

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


        self.photos=[]
        self.bboxes=[]
        self.objects=[]
        self.labels=[]
        self.img_height=[]
        self.img_width=[]
        cls = self.CATEGORIES
        self.class_to_idx = dict(zip(cls, range(len(cls))))
        self.idx_to_class = dict(zip(range(len(cls)), cls))
        
        file = open('/home/alouros/vscode/Drone_FasterRCNN/train_list.txt', 'r')
        lines= file.readlines()
        self.count = 0
        self.id_to_img_map = []
        for line in lines:
            splits=line.strip().split(",")
            self.photos.append(splits[0])
            self.img_width.append(splits[1])
            self.img_height.append(splits[2])
            self.objects.append(int(splits[3]))
            lbl=[]
            bbx=[]
            for obj in range(int(splits[3])):
                lbl.append(self.class_to_idx[splits[4 + obj * 5].strip()])
                bbx.append([int(splits[5 + obj * 5]),int(splits[6 + obj * 5]),int(splits[7 + obj * 5]),int(splits[8 + obj * 5])])
            self.labels.append(lbl)
            self.bboxes.append(bbx)
            self.id_to_img_map.append(self.count)
            self.count += 1

        #print('{} {}'.format(self.bboxes[1827], self.labels[1827] ))
        file.close()

        self.transforms = transforms


    def __getitem__(self, idx):
        # load the image as a PIL Image
        image =Image.open(self.photos[idx])

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
        return self.count

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
        print(boxlist.get_field("labels"))
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

