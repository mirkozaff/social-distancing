import torch, torchvision
from tqdm import tqdm
import os.path as path
import os
from transform import four_point_transform

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

if __name__ == '__main__':
    setup_logger()
 
    #create a detectron2 config and a detectron2 DefaultPredictor to run inference on this image.
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    images_path =  './video_frames/'
    images = [f for f in os.listdir(images_path) if path.isfile(path.join(images_path, f))]

    output_path = './predictions/'

    for im_name in tqdm(images):
        im = cv2.imread(path.join(images_path, im_name))
        outputs = predictor(im)

        # Filter out non-person labels
        pred_classes = outputs["instances"].pred_classes
        filtered = pred_classes == 0 # class 0 = person
        outputs["instances"] = outputs["instances"][filtered]
   
        # We can use `Visualizer` to draw the predictions on the image.
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        v = Visualizer(im[:, :, ::-1], metadata, scale=1.0, instance_mode = ColorMode.SEGMENTATION)
        #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Four perspective points in the following order: top-left, top-right, bottom-right, bottom-left
        pts = np.array([(353, 0), (913, 0), (1261, 656), (15, 668)])

        # Obtaining warped image and transformation matrix
        warped, M = four_point_transform(im, pts)
        cv2.imwrite('warped.png', warped)

        # Get all bounding box on cpu
        bboxes = outputs["instances"].pred_boxes.to("cpu")

        for instance in bboxes:
            # Get the bottom-left and bottom-right points of the bounding box in the trasnformed space
            a1 = np.dot(M, (instance[0], instance[3], 1))
            a1 = np.array([a1[0]/a1[2], a1[1]/a1[2]])
            a2 = np.dot(M, (instance[2], instance[3], 1))
            a2 = np.array([a2[0]/a2[2], a2[1]/a2[2]])

            # Calculate median point
            a = np.median(np.array([a1,a2]), axis=0)

            for instance2 in bboxes:
                # Get the bottom-left and bottom-right points of the bounding box in the trasnformed space
                b1 = np.dot(M, (instance2[0], instance2[3], 1))
                b1 = np.array([b1[0]/b1[2], b1[1]/b1[2]])
                b2 = np.dot(M, (instance2[2], instance2[3], 1))
                b2 = np.array([b2[0]/b2[2], b2[1]/b2[2]])

                # Calculate median point
                b = np.median(np.array([b1,b2]), axis=0)

                # Calculate distance between median points
                dist = np.linalg.norm(a - b)
                #dist = np.sqrt(((a[0]/a[2] - b[0]/b[2]) ** 2) + ((a[1]/a[2] - b[1]/b[2]) ** 2))

                if (dist < 90) and (dist != 0):
                    # Get median points in the original space
                    x1 = np.median(np.array([[instance[0], instance[3]], [instance[2], instance[3]]]), axis=0)
                    x2 = np.median(np.array([[instance2[0], instance2[3]], [instance2[2], instance2[3]]]), axis=0)

                    # Draw bounding boxes and connecting lines between median points
                    #x = v.draw_box(instance, alpha=0.5, edge_color='r', line_style='-')
                    #x = v.draw_box(instance2, alpha=0.5, edge_color='r', line_style='-')
                    x = v.draw_line([x1[0], x2[0]], [x1[1], x2[1]], color = 'r', linestyle="-", linewidth=None)

            x = v.draw_box(instance, alpha=0.5, edge_color='g', line_style='-')              

        # Write final image
        cv2.imwrite(path.join(output_path, im_name), x.get_image()[:, :, ::-1])