# Global Convolutional Network with PyTorch
A toy experiment of Global Convolutional Network given VOC2012 Datasets.
## Usage
```bash
git clone https://github.com/JamesHsu333/Global_Convolutional_Network.git
cd 
pip install -r requirements.txt
```
## Dataset
1. Download from Global_Convolutional_Network
[VOC2012 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
2. Extract directory ``` JPEGImages``` and ``` SegmentationClass``` under directory ```data/```
3. Extract ```train.txt``` and ```val.txt``` from ```VOC2012/ImageSets/Segmentation``` and put them under ```data/```
### Data Preprocessing
The images of VOC2012 are 500x225 pixels. Due to GPU memory limitations, they are resized to 224x224.
## Model Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,408
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
            Conv2d-4         [-1, 64, 112, 112]           4,096
       BatchNorm2d-5         [-1, 64, 112, 112]             128
              ReLU-6         [-1, 64, 112, 112]               0
            Conv2d-7         [-1, 64, 112, 112]          36,864
       BatchNorm2d-8         [-1, 64, 112, 112]             128
              ReLU-9         [-1, 64, 112, 112]               0
           Conv2d-10        [-1, 256, 112, 112]          16,384
      BatchNorm2d-11        [-1, 256, 112, 112]             512
           Conv2d-12        [-1, 256, 112, 112]          16,384
      BatchNorm2d-13        [-1, 256, 112, 112]             512
             ReLU-14        [-1, 256, 112, 112]               0
       Bottleneck-15        [-1, 256, 112, 112]               0
           Conv2d-16         [-1, 64, 112, 112]          16,384
      BatchNorm2d-17         [-1, 64, 112, 112]             128
             ReLU-18         [-1, 64, 112, 112]               0
           Conv2d-19         [-1, 64, 112, 112]          36,864
      BatchNorm2d-20         [-1, 64, 112, 112]             128
             ReLU-21         [-1, 64, 112, 112]               0
           Conv2d-22        [-1, 256, 112, 112]          16,384
      BatchNorm2d-23        [-1, 256, 112, 112]             512
             ReLU-24        [-1, 256, 112, 112]               0
       Bottleneck-25        [-1, 256, 112, 112]               0
           Conv2d-26         [-1, 64, 112, 112]          16,384
      BatchNorm2d-27         [-1, 64, 112, 112]             128
             ReLU-28         [-1, 64, 112, 112]               0
           Conv2d-29         [-1, 64, 112, 112]          36,864
      BatchNorm2d-30         [-1, 64, 112, 112]             128
             ReLU-31         [-1, 64, 112, 112]               0
           Conv2d-32        [-1, 256, 112, 112]          16,384
      BatchNorm2d-33        [-1, 256, 112, 112]             512
             ReLU-34        [-1, 256, 112, 112]               0
       Bottleneck-35        [-1, 256, 112, 112]               0
           Conv2d-36        [-1, 128, 112, 112]          32,768
      BatchNorm2d-37        [-1, 128, 112, 112]             256
             ReLU-38        [-1, 128, 112, 112]               0
           Conv2d-39          [-1, 128, 56, 56]         147,456
      BatchNorm2d-40          [-1, 128, 56, 56]             256
             ReLU-41          [-1, 128, 56, 56]               0
           Conv2d-42          [-1, 512, 56, 56]          65,536
      BatchNorm2d-43          [-1, 512, 56, 56]           1,024
           Conv2d-44          [-1, 512, 56, 56]         131,072
      BatchNorm2d-45          [-1, 512, 56, 56]           1,024
             ReLU-46          [-1, 512, 56, 56]               0
       Bottleneck-47          [-1, 512, 56, 56]               0
           Conv2d-48          [-1, 128, 56, 56]          65,536
      BatchNorm2d-49          [-1, 128, 56, 56]             256
             ReLU-50          [-1, 128, 56, 56]               0
           Conv2d-51          [-1, 128, 56, 56]         147,456
      BatchNorm2d-52          [-1, 128, 56, 56]             256
             ReLU-53          [-1, 128, 56, 56]               0
           Conv2d-54          [-1, 512, 56, 56]          65,536
      BatchNorm2d-55          [-1, 512, 56, 56]           1,024
             ReLU-56          [-1, 512, 56, 56]               0
       Bottleneck-57          [-1, 512, 56, 56]               0
           Conv2d-58          [-1, 128, 56, 56]          65,536
      BatchNorm2d-59          [-1, 128, 56, 56]             256
             ReLU-60          [-1, 128, 56, 56]               0
           Conv2d-61          [-1, 128, 56, 56]         147,456
      BatchNorm2d-62          [-1, 128, 56, 56]             256
             ReLU-63          [-1, 128, 56, 56]               0
           Conv2d-64          [-1, 512, 56, 56]          65,536
      BatchNorm2d-65          [-1, 512, 56, 56]           1,024
             ReLU-66          [-1, 512, 56, 56]               0
       Bottleneck-67          [-1, 512, 56, 56]               0
           Conv2d-68          [-1, 128, 56, 56]          65,536
      BatchNorm2d-69          [-1, 128, 56, 56]             256
             ReLU-70          [-1, 128, 56, 56]               0
           Conv2d-71          [-1, 128, 56, 56]         147,456
      BatchNorm2d-72          [-1, 128, 56, 56]             256
             ReLU-73          [-1, 128, 56, 56]               0
           Conv2d-74          [-1, 512, 56, 56]          65,536
      BatchNorm2d-75          [-1, 512, 56, 56]           1,024
             ReLU-76          [-1, 512, 56, 56]               0
       Bottleneck-77          [-1, 512, 56, 56]               0
           Conv2d-78          [-1, 256, 56, 56]         131,072
      BatchNorm2d-79          [-1, 256, 56, 56]             512
             ReLU-80          [-1, 256, 56, 56]               0
           Conv2d-81          [-1, 256, 28, 28]         589,824
      BatchNorm2d-82          [-1, 256, 28, 28]             512
             ReLU-83          [-1, 256, 28, 28]               0
           Conv2d-84         [-1, 1024, 28, 28]         262,144
      BatchNorm2d-85         [-1, 1024, 28, 28]           2,048
           Conv2d-86         [-1, 1024, 28, 28]         524,288
      BatchNorm2d-87         [-1, 1024, 28, 28]           2,048
             ReLU-88         [-1, 1024, 28, 28]               0
       Bottleneck-89         [-1, 1024, 28, 28]               0
           Conv2d-90          [-1, 256, 28, 28]         262,144
      BatchNorm2d-91          [-1, 256, 28, 28]             512
             ReLU-92          [-1, 256, 28, 28]               0
           Conv2d-93          [-1, 256, 28, 28]         589,824
      BatchNorm2d-94          [-1, 256, 28, 28]             512
             ReLU-95          [-1, 256, 28, 28]               0
           Conv2d-96         [-1, 1024, 28, 28]         262,144
      BatchNorm2d-97         [-1, 1024, 28, 28]           2,048
             ReLU-98         [-1, 1024, 28, 28]               0
       Bottleneck-99         [-1, 1024, 28, 28]               0
          Conv2d-100          [-1, 256, 28, 28]         262,144
     BatchNorm2d-101          [-1, 256, 28, 28]             512
            ReLU-102          [-1, 256, 28, 28]               0
          Conv2d-103          [-1, 256, 28, 28]         589,824
     BatchNorm2d-104          [-1, 256, 28, 28]             512
            ReLU-105          [-1, 256, 28, 28]               0
          Conv2d-106         [-1, 1024, 28, 28]         262,144
     BatchNorm2d-107         [-1, 1024, 28, 28]           2,048
            ReLU-108         [-1, 1024, 28, 28]               0
      Bottleneck-109         [-1, 1024, 28, 28]               0
          Conv2d-110          [-1, 256, 28, 28]         262,144
     BatchNorm2d-111          [-1, 256, 28, 28]             512
            ReLU-112          [-1, 256, 28, 28]               0
          Conv2d-113          [-1, 256, 28, 28]         589,824
     BatchNorm2d-114          [-1, 256, 28, 28]             512
            ReLU-115          [-1, 256, 28, 28]               0
          Conv2d-116         [-1, 1024, 28, 28]         262,144
     BatchNorm2d-117         [-1, 1024, 28, 28]           2,048
            ReLU-118         [-1, 1024, 28, 28]               0
      Bottleneck-119         [-1, 1024, 28, 28]               0
          Conv2d-120          [-1, 256, 28, 28]         262,144
     BatchNorm2d-121          [-1, 256, 28, 28]             512
            ReLU-122          [-1, 256, 28, 28]               0
          Conv2d-123          [-1, 256, 28, 28]         589,824
     BatchNorm2d-124          [-1, 256, 28, 28]             512
            ReLU-125          [-1, 256, 28, 28]               0
          Conv2d-126         [-1, 1024, 28, 28]         262,144
     BatchNorm2d-127         [-1, 1024, 28, 28]           2,048
            ReLU-128         [-1, 1024, 28, 28]               0
      Bottleneck-129         [-1, 1024, 28, 28]               0
          Conv2d-130          [-1, 256, 28, 28]         262,144
     BatchNorm2d-131          [-1, 256, 28, 28]             512
            ReLU-132          [-1, 256, 28, 28]               0
          Conv2d-133          [-1, 256, 28, 28]         589,824
     BatchNorm2d-134          [-1, 256, 28, 28]             512
            ReLU-135          [-1, 256, 28, 28]               0
          Conv2d-136         [-1, 1024, 28, 28]         262,144
     BatchNorm2d-137         [-1, 1024, 28, 28]           2,048
            ReLU-138         [-1, 1024, 28, 28]               0
      Bottleneck-139         [-1, 1024, 28, 28]               0
          Conv2d-140          [-1, 512, 28, 28]         524,288
     BatchNorm2d-141          [-1, 512, 28, 28]           1,024
            ReLU-142          [-1, 512, 28, 28]               0
          Conv2d-143          [-1, 512, 14, 14]       2,359,296
     BatchNorm2d-144          [-1, 512, 14, 14]           1,024
            ReLU-145          [-1, 512, 14, 14]               0
          Conv2d-146         [-1, 2048, 14, 14]       1,048,576
     BatchNorm2d-147         [-1, 2048, 14, 14]           4,096
          Conv2d-148         [-1, 2048, 14, 14]       2,097,152
     BatchNorm2d-149         [-1, 2048, 14, 14]           4,096
            ReLU-150         [-1, 2048, 14, 14]               0
      Bottleneck-151         [-1, 2048, 14, 14]               0
          Conv2d-152          [-1, 512, 14, 14]       1,048,576
     BatchNorm2d-153          [-1, 512, 14, 14]           1,024
            ReLU-154          [-1, 512, 14, 14]               0
          Conv2d-155          [-1, 512, 14, 14]       2,359,296
     BatchNorm2d-156          [-1, 512, 14, 14]           1,024
            ReLU-157          [-1, 512, 14, 14]               0
          Conv2d-158         [-1, 2048, 14, 14]       1,048,576
     BatchNorm2d-159         [-1, 2048, 14, 14]           4,096
            ReLU-160         [-1, 2048, 14, 14]               0
      Bottleneck-161         [-1, 2048, 14, 14]               0
          Conv2d-162          [-1, 512, 14, 14]       1,048,576
     BatchNorm2d-163          [-1, 512, 14, 14]           1,024
            ReLU-164          [-1, 512, 14, 14]               0
          Conv2d-165          [-1, 512, 14, 14]       2,359,296
     BatchNorm2d-166          [-1, 512, 14, 14]           1,024
            ReLU-167          [-1, 512, 14, 14]               0
          Conv2d-168         [-1, 2048, 14, 14]       1,048,576
     BatchNorm2d-169         [-1, 2048, 14, 14]           4,096
            ReLU-170         [-1, 2048, 14, 14]               0
      Bottleneck-171         [-1, 2048, 14, 14]               0
          Conv2d-172         [-1, 21, 112, 112]          37,653
          Conv2d-173         [-1, 21, 112, 112]           3,108
          Conv2d-174         [-1, 21, 118, 106]          37,653
          Conv2d-175         [-1, 21, 112, 112]           3,108
             GCN-176         [-1, 21, 112, 112]               0
          Conv2d-177         [-1, 21, 112, 112]           3,990
            ReLU-178         [-1, 21, 112, 112]               0
          Conv2d-179         [-1, 21, 112, 112]           3,990
              BR-180         [-1, 21, 112, 112]               0
          Conv2d-181           [-1, 21, 56, 56]          75,285
          Conv2d-182           [-1, 21, 56, 56]           3,108
          Conv2d-183           [-1, 21, 62, 50]          75,285
          Conv2d-184           [-1, 21, 56, 56]           3,108
             GCN-185           [-1, 21, 56, 56]               0
          Conv2d-186           [-1, 21, 56, 56]           3,990
            ReLU-187           [-1, 21, 56, 56]               0
          Conv2d-188           [-1, 21, 56, 56]           3,990
              BR-189           [-1, 21, 56, 56]               0
          Conv2d-190           [-1, 21, 28, 28]         150,549
          Conv2d-191           [-1, 21, 28, 28]           3,108
          Conv2d-192           [-1, 21, 34, 22]         150,549
          Conv2d-193           [-1, 21, 28, 28]           3,108
             GCN-194           [-1, 21, 28, 28]               0
          Conv2d-195           [-1, 21, 28, 28]           3,990
            ReLU-196           [-1, 21, 28, 28]               0
          Conv2d-197           [-1, 21, 28, 28]           3,990
              BR-198           [-1, 21, 28, 28]               0
          Conv2d-199           [-1, 21, 14, 14]         301,077
          Conv2d-200           [-1, 21, 14, 14]           3,108
          Conv2d-201            [-1, 21, 20, 8]         301,077
          Conv2d-202           [-1, 21, 14, 14]           3,108
             GCN-203           [-1, 21, 14, 14]               0
          Conv2d-204           [-1, 21, 14, 14]           3,990
            ReLU-205           [-1, 21, 14, 14]               0
          Conv2d-206           [-1, 21, 14, 14]           3,990
              BR-207           [-1, 21, 14, 14]               0
          Conv2d-208           [-1, 21, 28, 28]           3,990
            ReLU-209           [-1, 21, 28, 28]               0
          Conv2d-210           [-1, 21, 28, 28]           3,990
              BR-211           [-1, 21, 28, 28]               0
          Conv2d-212           [-1, 21, 56, 56]           3,990
            ReLU-213           [-1, 21, 56, 56]               0
          Conv2d-214           [-1, 21, 56, 56]           3,990
              BR-215           [-1, 21, 56, 56]               0
          Conv2d-216         [-1, 21, 112, 112]           3,990
            ReLU-217         [-1, 21, 112, 112]               0
          Conv2d-218         [-1, 21, 112, 112]           3,990
              BR-219         [-1, 21, 112, 112]               0
          Conv2d-220         [-1, 21, 112, 112]           3,990
            ReLU-221         [-1, 21, 112, 112]               0
          Conv2d-222         [-1, 21, 112, 112]           3,990
              BR-223         [-1, 21, 112, 112]               0
          Conv2d-224         [-1, 21, 224, 224]           3,990
            ReLU-225         [-1, 21, 224, 224]               0
          Conv2d-226         [-1, 21, 224, 224]           3,990
              BR-227         [-1, 21, 224, 224]               0
================================================================
Total params: 24,733,844
Trainable params: 24,733,844
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 1159.64
Params size (MB): 94.35
Estimated Total Size (MB): 1254.56
----------------------------------------------------------------
```
## Quickstart
1.  Created a ```base_model``` directory under the ```experiments``` directory. It contains a file ```params.json``` which sets the hyperparameters for the experiment. It looks like
```Json
{
    "learning_rate": 0.001,
    "batch_size": 5,
    "num_epochs": 35,
    "dropout_rate": 0.0,
    "num_channels": 32,
    "save_summary_steps": 100,
    "num_workers": 4
}
```
2. Train your experiment. Run
```bash
python train.py
```
3. Created a new directory ```learning_rate``` in experiments. Run
```bash
python search_hyperparams.py --parent_dir experiments/learning_rate
```
It will train and evaluate a model with different values of learning rate defined in ```search_hyperparams.py``` and create a new directory for each experiment under ```experiments/learning_rate/```.
4. Display the results of the hyperparameters search in a nice format
```bash
python synthesize_results.py --parent_dir experiments/learning_rate
```
5. Evaluation on the test set Once you've run many experiments and selected your best model and hyperparameters based on the performance on the validation set, you can finally evaluate the performance of your model on the test set. Run
```bash
python evaluate.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```
## Resources
* For more Project Structure details, please refer to [Deep Learning Project Structure](https://deeps.site/blog/2019/12/07/dl-project-structure/)
* Code implementation refers from [SConsul/Global_Convolutional_Network](https://github.com/SConsul/Global_Convolutional_Network)

## References
[[1]](https://arxiv.org/abs/1703.02719) Chao Peng, Xiangyu Zhang, Gang Yu, Guiming Luo, and Jian Sun. Large kernel matters - improve semantic segmentation by global convolutional network. CoRR, abs/1703.02719, 2017.