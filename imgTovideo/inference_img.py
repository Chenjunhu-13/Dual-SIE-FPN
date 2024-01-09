import torch
import numpy as np
import cv2
import os
from PIL import Image
# from matplotlib import pyplot
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, Metadata
# from bmaskrcnn import add_boundary_preserving_config
import matplotlib
matplotlib.use("Agg")

# 0701  0801  0802  1001
im_folder = "/data/datasets/COCO_UAVDT/UAVDT_M0403"

# save_folder = "F:/cjh/FE-FPN/box_results_fpn"
save_folder = "/data/datasets/COCO_UAVDT/UAVDT_Predict3"

cfg = get_cfg()
# add_boundary_preserving_config(cfg)
# cfg.merge_from_file('/data/cjh_datasets/cjh/projects/detectron2_test/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml')
cfg.merge_from_file('/data/cjh_datasets/cjh/projects/detectron2_test/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml')

# cfg.MODEL.WEIGHTS = "/data/cjh_datasets/uavdt_results/uavdt_result_faster_r101_125/model_best.pth"
# cfg.MODEL.WEIGHTS = "/data/cjh_datasets/model_weights/faster_rcnn_nearest_b16/faster_rcnn_r50_fpn_1x_nearest.pth"
cfg.MODEL.WEIGHTS = "/data/cjh_datasets/cjh/projects/detectron2_test/output/model_best.pth"


# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 模型阈值
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
CLASS_NAMES = ["car", "truck", "bus"]
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7  # 模型阈值
cfg.MODEL.RETINANET.NUM_CLASSES = 3

# dict = torch.load(cfg.MODEL.WEIGHTS)
# for k in dict['model'].keys():
#     print(k)

for im_file in os.listdir(im_folder):
    im = cv2.imread(os.path.join(im_folder, im_file))
    print(im.shape)
    save_result_path = os.path.join(save_folder, im_file)

    height = im.shape[0]
    width = im.shape[1]
    dpi = 500
    metadata = Metadata()
    metadata.thing_classes = ['car', 'truck', 'bus']
    metadata.thing_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    predictor = DefaultPredictor(cfg)
    predictor.metadata = metadata

    outputs = predictor(im)

    class_names = outputs["instances"].get("pred_classes").tolist()
    # print(metadata.thing_classes)
    # print(class_names)
    if class_names:
        print(metadata.thing_classes[class_names[0]])
    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode= ColorMode.SEGMENTATION)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    # 提取预测结果
    predictions = outputs["instances"].to("cpu")
    v = v.draw_instance_predictions(predictions)

    num_classes = metadata.get("thing_classes", None)

    print(num_classes)
    colors = metadata.get("thing_colors", None)
    print(colors)
    result = v.get_image()[:, :, ::-1]
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # 在原图上画出检测结果
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1, instance_mode=ColorMode.SEGMENTATION)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(result)

    # plt.imshow(v.get_image())
    plt.savefig(save_result_path)  # 保存结果

print('Image inference completed!!!')