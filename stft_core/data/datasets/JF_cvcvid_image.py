import os
import pickle

import torch
import torch.utils.data

from PIL import Image
import cv2
import sys
import numpy as np
from pathlib import Path

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from stft_core.structures.bounding_box import BoxList
from stft_core.utils.comm import is_main_process


class JF_CVCVIDImageDataset(torch.utils.data.Dataset):
    # TODO: background Klasse entfernen?
    # dann auch anpassen beim annotation laden
    classes = ['__background__',  # always index 0
                'hp',
                'ad']

    def __init__(self, image_set, data_dir, img_dir, anno_path, img_index, transforms=None, is_train=True):
        self.det_vid = image_set.split("_")[0]
        self.image_set = image_set
        self.transforms = transforms

        self.data_dir = data_dir
        self.img_dir = img_dir
        self.anno_path = anno_path
        # index file that gives the paths to every image to be used
        self.img_index = img_index

        self.is_train = is_train

        # set file formats for image and annotations
        img_file_type = ".jpg"
        self._img_dir = os.path.join(self.img_dir, "%s" + img_file_type)
        self._anno_path = os.path.join(self.anno_path, "%s.xml")

        with open(self.img_index) as f:
            lines = [x.strip().split(" ") for x in f.readlines()]

        # # give one dir per line and use all frames inside
        # if len(lines[0]) == 1:
        #
        #     # get all image files
        #     img_files = [f for line in lines for f in Path(line[0]).glob("*/*" + img_file_type)]
        #
        #     self.frames_vidDir_and_filename = [Path(path.parent.absolute().name + '/' + path.stem) for path in img_files] #case1/case_M_20181001100941_0U62372100109341_1_005_001-1_a2_ayy_image0001
        #     self.frames_vidDir_and_filename_pattern = [x[0] + "/" + x[0].split('-')[0] + "-%d" for x in lines] #12-5/12-%d
        #     # list of all frame ids given in the index file
        #     # not done yet
        #     self.frame_id = [int(x[1]) for x in lines]
        #     self.frame_seg_id = [int(x[2]) for x in lines]
        #     self.frame_seg_len = [int(x[3]) for x in lines]

        # if only two entries per line are given instead of 4
        if len(lines[0]) == 2:
            self.frames_vidDir_and_filename = [x[0] for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
        else:
            self.frames_vidDir_and_filename = ["%s/%d" % (x[0], int(x[2])) for x in lines] #case4/13
            self.frames_vidDir_and_filename_pattern = [x[0] + "/" + "%d" for x in lines] #12-5/12-%d
            # list of all frame ids given in the index file
            self.frame_id = [int(x[1]) for x in lines]
            self.frame_seg_id = [int(x[2]) for x in lines]
            self.frame_seg_len = [int(x[3]) for x in lines]



        self.annos = self.load_annos(os.path.join(self.cache_dir, self.image_set + "_anno.pkl"))

        if self.is_train:
            print('Loaded  Training  set : {} , number samples: {}'.format(anno_path, len(self.frames_vidDir_and_filename)))
        else:
            print('Loaded  Validation  set : {} , number samples: {}'.format(anno_path, len(self.frames_vidDir_and_filename)))


    def __getitem__(self, idx):
        if self.is_train:
            return self._get_train(idx)
        else:
            return self._get_test(idx)

    def _get_train(self, idx):
        filename = self.frames_vidDir_and_filename[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def _get_test(self, idx):
        return self._get_train(idx)


    def __len__(self):
        return len(self.frames_vidDir_and_filename)

    @property
    def cache_dir(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_dir = os.path.join(self.data_dir, 'cache')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir


    # actually processes the annotation for each file
    # extracts the info from the xml and converts to array
    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        objs = target.findall("object")
        for obj in objs:
            bbox =obj.find("bndbox")
            box = [
                np.maximum(float(bbox.find("xmin").text), 0),
                np.maximum(float(bbox.find("ymin").text), 0),
                np.minimum(float(bbox.find("xmax").text), im_info[1] - 1),
                np.minimum(float(bbox.find("ymax").text), im_info[0] - 1)
            ]
            boxes.append(box)



            # find class in xml
            class_name = obj.find("name").text


            # map found class to the classes array from this dataloader class
            # hp == 1
            # ad == 2
            # KUMC dataset
            if class_name == "adenomatous":
                class_int = 2
            elif class_name == "hyperplastic":
                class_int = 1
            # SUN dataset
            elif class_name in ["Low-grade adenoma", "High-grade adenoma", "Traditional serrated adenoma", "Invasive cancer (T1b)"]:
                class_int = 2
            elif class_name in ["Hyperplastic polyp", "Sessile serrated lesion"]:
                class_int = 1


            gt_classes.append(class_int)

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(gt_classes),
            "im_info": im_info,
        }
        return res

    # load the annotation of each file
    # example original: [{'boxes': tensor([[129.5000, 244.5000, 197.5000, 312.5000]]), 'labels': tensor([1]), 'im_info': (480, 608)}, {'boxes':....
    def load_annos(self, cache_file):
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                annos = pickle.load(fid)
            if is_main_process():
                print("{}'s annotation information loaded from {}".format(self.det_vid, cache_file))
        else:
            annos = []
            for idx in range(len(self)):
                if idx % 1000 == 0:
                    print("Had processed {} images".format(idx))

                filename = self.frames_vidDir_and_filename[idx]

                tree = ET.parse(self._anno_path % filename).getroot()
                anno = self._preprocess_annotation(tree)
                annos.append(anno)
            print("Had processed {} images".format(len(self)))

            if is_main_process():
                with open(cache_file, "wb") as fid:
                    pickle.dump(annos, fid)
                print("Saving {}'s annotation information into {}".format(self.det_vid, cache_file))

        return annos

    def get_img_info(self, idx):
        im_info = self.annos[idx]["im_info"]
        return {"height": im_info[0], "width": im_info[1]}


    def get_img_name(self, idx):
        filename = self.frames_vidDir_and_filename[idx]
        return filename

    def get_visualization(self, idx):
        filename = self.frames_vidDir_and_filename[idx]

        img = cv2.imread(self._img_dir % filename)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        return img, target, filename

    def get_groundtruth(self, idx):
        anno = self.annos[idx]

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])

        return target

    @staticmethod
    def map_class_id_to_class_name(class_id):
        return JF_CVCVIDImageDataset.classes[class_id]
