import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

np.set_printoptions(threshold=np.inf)

# Decode Segmentation Mask
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

dataDir='../../Dataset/COCO'
dataType='train2017'
annFile='{}/train2017/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
catIds = coco.getCatIds(catNms=['airplane','bicycle','bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 'couch', 'train', 'tv'])
imgIds = coco.getImgIds(catIds=catIds[0])
print(len(imgIds))
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

print(img)

I = Image.open(os.path.join("{}/train2017/train2017/{}".format(dataDir, img['file_name'])))
# load and display instance annotations
plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds[0], iscrowd=None)
anns = coco.loadAnns(annIds)
print(anns)
coco.showAnns(anns)
plt.show()
"""
data = os.path.join("../../Dataset/VOC2012/VOC2012/SegmentationClass/2007_000121.png")
mask = Image.open(data)
mask = np.array(mask)
mask = np.where(mask==255,0,mask)
plt.subplot(1, 3, 1)
plt.title('label')
plt.imshow(decode_segmap(mask))
plt.axis('off')
plt.show()
"""