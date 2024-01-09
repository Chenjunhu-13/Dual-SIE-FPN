import matplotlib
import matplotlib.pyplot as plt
from detectron2_gradcam import Detectron2GradCAM

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

from gradcam import GradCAM, GradCamPlusPlus
import cv2


plt.rcParams["figure.figsize"] = (30,10)

img_path = "img/000000243123.jpg"

img1 = cv2.imread(img_path)
# config-file
config_file = "F:/cjh/Code_py/Projects/detectron2_test/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
# trained pth file
model_file = "F:/cjh/FE-FPN/COCO_train_results/faster_rcnn_nearest_b16/faster_rcnn_r50_fpn_1x_nearest.pth"
# model_file = "F:/cjh/FE-FPN/COCO_train_results/lff_nlup_rf_b16/model_final.pth"

config_list = [
"MODEL.ROI_HEADS.SCORE_THRESH_TEST", "0.5",
"MODEL.ROI_HEADS.NUM_CLASSES", "80",
"MODEL.WEIGHTS", model_file
]

layer_name = "backbone.bottom_up.res5.2.conv3"
instance = 4 #CAM is generated per object instance, not per class!

def main():
    cam_extractor = Detectron2GradCAM(config_file, config_list, img_path=img_path)
    grad_cam = GradCamPlusPlus

    image_dict0, cam_orig0 = cam_extractor.get_cam(target_instance=0, layer_name=layer_name, grad_cam_instance=grad_cam)
    instances = len(image_dict0["output"]["instances"].scores)
    print(instances)

    image_dict_cam = 0
    for instance in range(0, instances):
        image_dict, cam_orig = cam_extractor.get_cam(target_instance=instance, layer_name=layer_name, grad_cam_instance=grad_cam)
        image_dict_cam += image_dict["cam"]
    print(image_dict_cam)

    # image_dict, cam_orig = cam_extractor.get_cam(target_instance=1, layer_name=layer_name, grad_cam_instance=grad_cam)


    # v = Visualizer(image_dict["image"], MetadataCatalog.get(cam_extractor.cfg.DATASETS.TRAIN[0]), scale=1.0)
    # out = v.draw_instance_predictions(image_dict["output"]["instances"][instance].to("cpu"))

    h = img1.shape[0]
    w = img1.shape[1]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    axes.imshow(img1, interpolation=None)
    axes.imshow(image_dict_cam, cmap='jet', alpha=0.5)
    #
    plt.savefig(f"instance_{instances}_cam_fpn.jpg", dpi=dpi)
    plt.show()


if __name__ == "__main__":
    main()
