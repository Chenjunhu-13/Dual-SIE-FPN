import os
import cv2

path = '/data/datasets/COCO_UAVDT/UAVDT_Predict3/'
filelist = os.listdir(path)

filelist = sorted(filelist)
#
# for file in filelist:
#     print(file)
fps = 20
save_path = '/data/datasets/COCO_UAVDT/UAVDT_403_VEDIO/'


file_path = save_path + 'saveVideo_{}.mp4'.format(403)
size = (1024, 540)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(file_path, fourcc, fps, size)

for item in filelist:
    print(item)
    if item.endswith('.jpg'):
        item = path + item
        img = cv2.imread(item)
        videoWriter.write(img)


videoWriter.release()
