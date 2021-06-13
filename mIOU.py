import os

import numpy as np
from PIL import Image
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        for gt, pre in zip(gt_image, pre_image):
            assert gt.shape == pre.shape
            self.confusion_matrix += self._generate_matrix(gt.flatten(), pre.flatten())

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def encode_segmap(image, nc=21):
    output = np.zeros(image.shape[:2]).astype(np.int32)
    labels = []
    label_colors = [(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

    for l in range(0, nc):
        labels.append(np.all(image == label_colors[l], axis=-1))
    for l in range(0, nc):
        if (~labels[l]).all():
            continue
        output += l*labels[l]
    return output

ex = np.array([[[0,0,0],[128,0,0]],[[0,0,0],[128,128,0]]], dtype = int)
print(encode_segmap(ex))

e = Evaluator(21)
e.reset()

mask_files = os.listdir("images\mask")
mask_files = [os.path.join("images\mask", f) for f in mask_files if f.endswith('.png')]

prediction_files = os.listdir("images\prediction")
prediction_files = [os.path.join("images\prediction", f) for f in prediction_files if f.endswith('.png')]

with tqdm(total=len(mask_files)) as t:
    for mask, pre in zip(mask_files, prediction_files):
        mask = Image.open(mask)
        pre = Image.open(pre)

        mask = np.array(mask)
        pre = np.array(pre)
        mask = encode_segmap(mask)
        pre = encode_segmap(pre)

        e.add_batch(mask, pre)
        t.update()



"""
a = np.arange(36).reshape(4,3,3) + 10
a = np.argmax(a, axis=0)
print(a.shape)

label = np.array([[[0, 1 ,0], [1, 1, 0], [2, 3, 3]]], dtype = int)
predict = np.array([[[[1, 1 ,1], [0, 0, 0], [0, 0, 1]],[[0, 0 ,0], [1, 1, 0], [0, 0, 0]],[[0, 0 ,0], [0, 0, 1], [1, 0, 0]],[[0, 0 ,0], [0, 0, 0], [0, 1, 0]]]], dtype = int)
#[[0, 0 ,0], [1, 1, 2], [2, 3, 0]]
print(label.shape)
print(predict.shape)
print(predict)
m = net.metrics
m1 = m['mIOU'](predict, label, 4)
e = net.Evaluator(4)

e.reset()

e.add_batch(label, predict)
predict_1 = np.array([[[[1, 1 ,0], [0, 0, 0], [0, 0, 1]],[[0, 0 ,0], [1, 1, 0], [0, 0, 0]],[[0, 0 ,0], [0, 0, 1], [0, 0, 0]],[[0, 0 ,1], [0, 0, 0], [1, 1, 0]]]], dtype = int)
#[[0, 0 ,3], [1, 1, 2], [3, 3, 0]]
# e.reset()
e.add_batch(label, predict_1)
m2 = m['mIOU'](predict_1, label, 4)

print(m1)
print(m2)
print((m1+m2)/2)
"""
mIOU = e.Mean_Intersection_over_Union()

print(mIOU)