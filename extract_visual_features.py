import argparse
import distutils.core
import glob
import json
import os
import sys
from os.path import join as opj

import natsort

import detectron2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.boxes import Boxes

from dataset import OBJECTS, crawl_dataset

import cv2

# from detectron2.utils.logger import setup_logger
# setup_logger()




# from detectron2.structures.instances import Instances



class ObjectFeatureExtractor:
    def __init__(self, device='cuda:0'):
        config_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        model_path = config_path

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_path))
        
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.predictor = DefaultPredictor(self.cfg)
        self.device = device

    @torch.no_grad()
    def extract_features(self, image, bboxes, return_pred=False):
        # Partially based on ASSIGN paper's code
        # I also use model.roi_heads.box_head

        bboxes = Boxes(torch.from_numpy(bboxes).to(self.device))

        raw_height, raw_width = image.shape[:2]

        # Pre-processing
        image = self.predictor.aug.get_transform(image).apply_image(image)

        # Scale boxes
        new_height, new_width = image.shape[:2]
        scale_x = new_width / raw_width
        scale_y = new_height / raw_height
        bboxes.scale(scale_x=scale_x, scale_y=scale_y)


        image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
        inputs = [{'image': image, 'height': raw_height, 'width': raw_width}]
        images = self.predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = self.predictor.model.backbone(images.tensor)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [bboxes]
        features = [features[feature_name] for feature_name in self.predictor.model.roi_heads.in_features]
        box_features = self.predictor.model.roi_heads.box_pooler(features, proposal_boxes)
        
        roi_features=self.predictor.model.roi_heads.box_head(box_features) # This is what I prefered
        # roi_features = self.box_features.mean(dim=[2, 3])  # This is what ASSIGN does

        if return_pred:
            ret=self.predictor.model.roi_heads.box_predictor(roi_features)
            # ret[0].shape: (num_objects, 81) 80 classes + 1 background
            # ret[1].shape: (num_objects, 320) 80 classes x 4 coordinates        
            return roi_features, ret[0]
    
        else:
            return roi_features


def get_bbox_from_3d(bbox_3d):
    # Hands do not have 2d bboxes, so we need to convert 3d bboxes to 2d bboxes
    ...

def convert_box_format():
    ...

def main(args):

    # Extract features
    extractor = ObjectFeatureExtractor()

    # create dir to save features
    raw_dir = opj(args.root, "raw")
    vis_feat_dir = opj(raw_dir, "vis_feat")
    os.makedirs(vis_feat_dir, exist_ok=True)

    imgs_dir = args.imgs     

    for subject, task, take in crawl_dataset():
        
        rgb_files = natsort.natsorted(glob.glob(opj(imgs_dir, subject, task, take, "rgb", "*.png")))
        bboxes_files = natsort.natsorted(glob.glob(opj(imgs_dir, subject, task, take, "bboxes", "*.json")))
        
        bboxes_3d_files = natsort.natsorted(glob.glob(opj(imgs_dir, subject, task, take, "bboxes", "*.json")))
        
        assert len(rgb_files) == len(bboxes_files), "Number of rgb files and bboxes files must be the same"
        assert len(rgb_files) == len(bboxes_3d_files), "Number of rgb files and 3d bboxes files must be the same"

        for rgb_f, bboxes_f, bboxes_3d_f in zip(rgb_files, bboxes_files, bboxes_3d_files):
            
            image = cv2.imread(rgb_f)
            
            with open(bboxes_f, "r") as f:
                bboxes = json.load(f)
            
            bboxes_np = np.zeros((len(bboxes), 4))
            for oi, obj in enumerate(bboxes):
                bboxes_np[oi] = ...

            bboxes = convert_box_format(bboxes)

            with open(bboxes_3d_f, "r") as f:
                bboxes_3d = json.load(f)            
            
            obj_names = [obj["class_name"] for obj in bboxes_3d]

            features = extractor.extract_features(image, bboxes)

            torch.save(features, "...")
            # save object names (read it from 3d bboxes file)


        

def get_features_objectwise():
    # To make clustering of objects

    object_features = {obj : [] for obj in OBJECTS}

    for subject, task, take in crawl_dataset():
        feature_list = ...
        obj_name_list = ...

        for obj_name, feature in zip(obj_name_list, feature_list):
            object_features[obj_name].append(feature)


    return object_features

    

if __name__ == "__main__":
    
    imgs_default = opj(".", "raw", "bimacs_rgbd_data")

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-r", "--root", type=str, default=".", help="Bimanual dataset root directory")
    arg_parser.add_argument("--imgs", type=str, default=imgs_default, help="Bimanual dataset imgs directory")

    args = arg_parser.parse_args()

    main(args)