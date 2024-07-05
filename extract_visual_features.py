import argparse
import distutils.core
import glob
import json
import os
import sys
from os.path import join as opj


import natsort

import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

from dataset import OBJECTS, crawl_dataset

from PIL import Image


# Camera intrinsics
IMG_WIDTH=640
IMG_HEIGHT=480
FOV_X=54 / 180 * np.pi 
FOV_Y=45 / 180 * np.pi 

FX= IMG_WIDTH / (2 * np.tan(FOV_X/2)) # 628.035
FY= IMG_HEIGHT / (2 * np.tan(FOV_Y/2)) # 579.411
CX=IMG_WIDTH/2
CY=IMG_HEIGHT/2


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
    def extract_features(self, image, bboxes, return_early_feat=False, return_pred=False):
        # Returns roi_features, early_roi_features (optional), pred (optional)
        # Partially based on ASSIGN paper's code
        # I additionally use model.roi_heads.box_head

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
        
        if return_early_feat:
            bf_clone = box_features.clone() # This may not be necessary
            early_roi_features = bf_clone.mean(dim=[2, 3])  # This is what ASSIGN does

        roi_features=self.predictor.model.roi_heads.box_head(box_features) # This is what I prefered

        if return_pred:
            ret=self.predictor.model.roi_heads.box_predictor(roi_features)
            # ret[0].shape: (num_objects, 81) 80 classes + 1 background
            # ret[1].shape: (num_objects, 320) 80 classes x 4 coordinates        


        return_list = [roi_features]

        if return_early_feat:
            return_list.append(early_roi_features)

        if return_pred:
            return_list.append(ret[0])
        

        return return_list


def getRotationMatrix(w0: np.ndarray, theta: float = None):
    """Calculates rotation matrix using Rodrigue's formula (norm of the w is assigned to theta), (theta in radian)"""
    # FIXME modifies the w implicitly

    if theta is None:
        norm = np.linalg.norm(w0)
        w0 = w0 / norm
        theta = norm
    else:
        assert np.allclose(np.linalg.norm(w0), 1.0)

    W_hat = np.array([[0, -w0[2], w0[1]], [w0[2], 0, -w0[0]], [-w0[1], w0[0], 0]])

    # Rodrigue's formula
    R = (np.identity(3) + np.sin(theta) * W_hat + (1 - np.cos(theta)) * np.matmul(W_hat, W_hat))

    # sanity check
    assert np.allclose(np.linalg.det(R), 1.0)

    return R


def homogeneousMultiplication(A, x):
    """A.shape = [D, D], x.shape = [N, D-1];
    Calculates x' = A @ [x.T, 1].T
    then returns x' in inhomogeneous vector form
    """
    assert A.shape[0] == A.shape[1] and x.shape[1] == (A.shape[0] - 1)

    x = np.concatenate((x, np.ones((x.shape[0], 1))), 1)
    x = A @ x.T
    x = x[:-1] / x[-1]

    return x.T


def convert_from_xyz_and_rotate(x, y, z, angle, is_np=False):
    """
    x, y, z -> u, v, d
    """
    # Camera intrinsics
    #  uvd -> xyz

    if is_np:
        assert np.isinf(x).any() == False and np.isinf(y).any() == False and np.isinf(z).any() == False
        assert np.isnan(x).any() == False and np.isnan(y).any() == False and np.isnan(z).any() == False
    else:
        raise NotImplementedError("what are you doing?")    

    # rotate x,y,z point around x-axis 19 degree
    R=getRotationMatrix(np.array([1, 0, 0]), +np.pi*angle/180)
    A=np.identity(4)
    A[:3, :3]=R

    xyz=np.stack([x, y, z]).T

    rotated_xyz=homogeneousMultiplication(A, xyz)


    x, y, z = np.hsplit(rotated_xyz, 3)
    
    x, y, z = x.flatten(), y.flatten(), z.flatten()

    u = -FX * x / z + CX
    v = FY * y / z + CY

    mask_to_nan = (u < 0) | (u > IMG_WIDTH) | (v < 0) | (v > IMG_HEIGHT)     
    # print("u:", u)
    # print("v:", v)

    u[mask_to_nan] = np.nan
    v[mask_to_nan] = np.nan

    if is_np:
        return np.stack((u, v), axis=1)
    else:
        return u, v


def get_bbox_from_3d(bbox_3d, angle):
    # Hands do not have 2d bboxes, so we need to convert 3d bboxes to 2d bboxes
    bbox_3d_np=np.array([bbox_3d["x0"], bbox_3d["x1"], bbox_3d["y0"], bbox_3d["y1"], bbox_3d["z0"], bbox_3d["z1"]])
    
    vertices_xyz = np.array([[bbox_3d_np[0], bbox_3d_np[2], bbox_3d_np[4]], 
                             [bbox_3d_np[0], bbox_3d_np[2], bbox_3d_np[5]], 
                             [bbox_3d_np[0], bbox_3d_np[3], bbox_3d_np[4]], 
                             [bbox_3d_np[0], bbox_3d_np[3], bbox_3d_np[5]], 
                             [bbox_3d_np[1], bbox_3d_np[2], bbox_3d_np[4]], 
                             [bbox_3d_np[1], bbox_3d_np[2], bbox_3d_np[5]], 
                             [bbox_3d_np[1], bbox_3d_np[3], bbox_3d_np[4]], 
                             [bbox_3d_np[1], bbox_3d_np[3], bbox_3d_np[5]]])

    # This function also rotates the 3d bbox
    vertices_uv = convert_from_xyz_and_rotate(vertices_xyz[:, 0], vertices_xyz[:, 1], vertices_xyz[:, 2], angle, is_np=True)
    
    x0 = np.min(vertices_uv[:, 0])
    x1 = np.max(vertices_uv[:, 0])
    y0 = np.min(vertices_uv[:, 1])
    y1 = np.max(vertices_uv[:, 1])

    return np.array([x0, y0, x1, y1])

def convert_box_format(bbox):
    # bbox: (cx, cy, w, h) (0-1) -> (x1, y1, x2, y2) (0-width or 0-height)

    new_bbox = np.zeros_like(bbox)
    
    new_bbox[:, 0] = (bbox[:, 0] - bbox[:, 2] / 2) * IMG_WIDTH
    new_bbox[:, 1] = (bbox[:, 1] - bbox[:, 3] / 2) * IMG_HEIGHT
    new_bbox[:, 2] = (bbox[:, 0] + bbox[:, 2] / 2) * IMG_WIDTH
    new_bbox[:, 3] = (bbox[:, 1] + bbox[:, 3] / 2) * IMG_HEIGHT
    
    return new_bbox
    

def extend_bbox(bbox, extend_amount=0):
    # bbox: (x1, y1, x2, y2)
    # extend bbox by extend_amount
    bbox[0] = max(0, bbox[0] - extend_amount)
    bbox[1] = max(0, bbox[1] - extend_amount)
    bbox[2] = min(IMG_WIDTH, bbox[2] + extend_amount)
    bbox[3] = min(IMG_HEIGHT, bbox[3] + extend_amount)
    
    return bbox

def get_camera_angle(record_idx):
    # return camera angle for record_idx
    # TODO: unit test

    key_indices= [0, 3, 21, 90, 180, 201, 270, 360, 600]
    map= {
        "0": {"angle": -1}, # dummy
        "3": {"angle": 15},
        "21": {"angle": 21},
        "90": {"angle": 17.5},
        "180": {"angle": 20},
        "201": {"angle": 19},
        "270": {"angle": 18},
        "360": {"angle": 18},
        "600": {"angle": 16}
    }


    for i, key_idx in enumerate(key_indices):
        if record_idx < key_idx:
            break
    
    return map[str(key_indices[i])]["angle"]


def get_hands_bb_from_3d_obj(obj_3d):
    left_hand = None
    right_hand = None

    for obj in obj_3d:
        if obj["class_name"] == "LeftHand":
            left_hand = obj
        elif obj["class_name"] == "RightHand":
            right_hand = obj
    
    if left_hand is not None:
        left_bb=get_bbox_from_3d(left_hand["bounding_box"])
    else:
        left_bb=None

    if right_hand is not None:
        right_bb=get_bbox_from_3d(right_hand["bounding_box"])
    else:
        right_bb=None

    return left_bb, right_bb


def get_all_bb_from_3d_obj(obj_3d, angle):
    bboxes_np = np.zeros((len(obj_3d), 4))

    for oi, obj in enumerate(obj_3d):
        bbox=obj["bounding_box"]
        bboxes_np[oi] = get_bbox_from_3d(bbox, angle)

    return bboxes_np


def main(args):

    EXTRACT_EARLY_FEATURES_TOO = True

    # Extract features
    extractor = ObjectFeatureExtractor()

    # create dir to save features
    raw_dir = opj(args.root, "raw")
    vis_feat_dir = opj(raw_dir, "vis_feat")

    os.makedirs(vis_feat_dir, exist_ok=True)

    imgs_dir = args.imgs

    record_idx = 0
    prev_subject = ""
    for subject, task, take in crawl_dataset():
        
        if prev_subject != subject:
            prev_subject = subject
            print(f"Processing {subject}")

        print(".", end="", flush=True)

        rgb_files = natsort.natsorted(glob.glob(opj(imgs_dir, subject, task, take, "rgb", "chunk_*", "*.png")))
        # obj_2d_files = natsort.natsorted(glob.glob(opj(args.root, "raw", "bimacs_derived_data", subject, task, take, "2d_objects", "*.json")))
        obj_3d_files = natsort.natsorted(glob.glob(opj(args.root, "raw", "bimacs_derived_data", subject, task, take, "3d_objects", "*.json")))
        
        assert len(rgb_files) > 0, "No rgb files found"
        # assert len(rgb_files) == len(obj_2d_files), "Number of rgb files and bboxes files must be the same"
        assert len(rgb_files) == len(obj_3d_files), "Number of rgb files and 3d bboxes files must be the same"

        camera_angle = get_camera_angle(record_idx)
        target_dir = opj(vis_feat_dir, subject, task, take)
        os.makedirs(target_dir, exist_ok=True)

        frame_idx = 0

        # for rgb_f, obj_2d_f, obj_3d_f in zip(rgb_files, obj_2d_files, obj_3d_files):
        for rgb_f, obj_3d_f in zip(rgb_files, obj_3d_files):
            
            image = np.array(Image.open(rgb_f))
            if image.shape[-1] == 4:
                print(f"Warning: {rgb_f} has 4 channels. Dropping alpha channel")
                image = image[:, :, :3]

            
            # with open(obj_2d_f, "r") as f:
            #     obj_2d = json.load(f)
            
            # bboxes_np = np.zeros((len(obj_2d), 4))
            # for oi, obj in enumerate(obj_2d):
            #     bbox=obj["bounding_box"]
            #     bboxes_np[oi] = np.array([bbox["x"], bbox["y"], bbox["w"], bbox["h"]])

            # bboxes_np = convert_box_format(bboxes_np)

            with open(obj_3d_f, "r") as f:
                bboxes_3d = json.load(f)

            # lefthand_bb, righthand_bb = get_hands_bb_from_3d_obj(bboxes_3d)
            # concat_list = [bboxes_np]
            # if lefthand_bb is not None:
            #     concat_list.append(lefthand_bb)
            # if righthand_bb is not None:
            #     concat_list.append(righthand_bb)


            # bboxes_np = np.concatenate(concat_list, axis=0)
            # bboxes_np = extend_bbox(bboxes_np)
                
            # # # Alternative: use 3d bboxes to get 2d bboxes
            bboxes_np = get_all_bb_from_3d_obj(bboxes_3d, camera_angle)

            # CHECKME: if the order is correct
            obj_names = [obj["class_name"] for obj in bboxes_3d]

            features, early_features = extractor.extract_features(image, bboxes_np, return_early_feat=EXTRACT_EARLY_FEATURES_TOO, return_pred=False)

            torch.save(features, opj(target_dir, f"feat_{frame_idx:04d}.pt"))

            torch.save(early_features, opj(target_dir, f"early_feat_{frame_idx:04d}.pt"))

            # save object names (read it from 3d bboxes file)
            with open(opj(target_dir, f"objs_{frame_idx:04d}.txt"), "w") as f:
                f.write("\n".join(obj_names))
            
            frame_idx += 1
        
        # End of frame loop

        record_idx += 1

    # End of dataset loop



def get_features_objectwise(root, use_single_take=True, downsample=5):
    # To make clustering or other object-wise analysis

    raw_dir = opj(root, "raw")
    vis_feat_dir = opj(raw_dir, "vis_feat")

    object_features_dict = {obj : [] for obj in OBJECTS}
    early_object_features_dict = {obj : [] for obj in OBJECTS}

    for subject, task, take in crawl_dataset():
        if use_single_take:
            if take != "take_3":
                continue

        early_feature_list = natsort.natsorted(glob.glob(opj(vis_feat_dir, subject, task, take, "early_feat_*.pt")))
        feature_list = natsort.natsorted(glob.glob(opj(vis_feat_dir, subject, task, take, "feat_*.pt")))
        obj_file_list = natsort.natsorted(glob.glob(opj(vis_feat_dir, subject, task, take, "objs_*.txt")))

        early_feature_list = early_feature_list[::downsample]
        feature_list = feature_list[::downsample]
        obj_file_list = obj_file_list[::downsample]

        assert len(early_feature_list) == len(feature_list) and len(feature_list) == len(obj_file_list), "Number of files must be the same"

        for early_feature_file, feature_file, obj_file in zip(early_feature_list, feature_list, obj_file_list):
            
            eff_idx = early_feature_file.split("_")[-1][:4]
            ff_idx = feature_file.split("_")[-1][:4]
            of_idx = obj_file.split("_")[-1][:4]

            assert eff_idx == ff_idx and ff_idx == of_idx, "Index mismatch"

            with open(obj_file, "r") as f:
                obj_names = f.read().splitlines()

            features = torch.load(feature_file)
            early_features = torch.load(early_feature_file)

            for obj, feature, early_feature in zip(obj_names, features, early_features):
                object_features_dict[obj].append(feature)
                early_object_features_dict[obj].append(early_feature)


    for obj in object_features_dict:
        object_features_dict[obj] = torch.stack(object_features_dict[obj])
        early_object_features_dict[obj] = torch.stack(early_object_features_dict[obj])

    print("Total number of objects: ", sum([len(object_features_dict[obj]) for obj in object_features_dict]))

    return object_features_dict, early_object_features_dict

    

if __name__ == "__main__":
    

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-r", "--root", type=str, default=".", help="Bimanual dataset root directory")
    
    imgs_default = opj(".", "raw", "bimacs_rgbd_data")
    arg_parser.add_argument("--imgs", type=str, default=imgs_default, help="Bimanual dataset imgs directory")

    args = arg_parser.parse_args()
    
    assert os.path.exists(args.root), "Root directory does not exist"
    assert os.path.exists(args.imgs), "Images directory does not exist"

    main(args)