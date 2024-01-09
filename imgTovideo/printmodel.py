import torch

model = torch.load(r'/data/cjh_datasets/coco_weights/faster_rcnn_r50_fefpn_1x.pth')
print(model.keys())
for i in model['model']:
    print(i)