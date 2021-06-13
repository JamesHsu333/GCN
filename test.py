import argparse
import itertools 
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt
import model.net as net
import model.data_loader as data_loader
from PIL import Image

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../../Dataset/VOC2012/JPEGImages',
                    help="Directory containing the dataset")
parser.add_argument('--mask_dir', default='../../Dataset/VOC2012/SegmentationClass',
                    help="Directory containing the mask dataset")
parser.add_argument('--dataset_dir', default='../../Dataset/VOC2012/ImageSets/Segmentation',
                    help="Directory containing the train/val/test file names")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--model_type', default='GCN_Resnet',
                    help="Type of GCN")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--show_images', default='no', help="Show image")
parser.add_argument('--num_classes', default=21,
                    help="Numbers of classes")

def evaluate_save_images(model, loss_fns, dataloader, params):
    model.eval()
    i=1
    for _, (data_batch, labels_batch) in enumerate(dataloader):
        with torch.no_grad():
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(
                non_blocking=True), labels_batch.cuda(non_blocking=True)

                labels_batch=labels_batch.long()
                    
                output_batch = model(data_batch).float()

                loss = loss_fns['CrossEntropy'](output_batch, labels_batch)

                output_batch = output_batch.data.cpu().numpy()
                data_batch = data_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # Save image
                save_images(labels_batch, output_batch, i)
                i+=1
    return

def evaluate_find(model, loss_fns, dataloader, metrics, model_type, params):
    model.eval()

    summ = []
    plot = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            with torch.no_grad():
                if params.cuda:
                    data_batch, labels_batch = data_batch.cuda(
                        non_blocking=True), labels_batch.cuda(non_blocking=True)

                labels_batch=labels_batch.long()
                
                output_batch = model(data_batch).float()

                loss = loss_fns['CrossEntropy'](output_batch, labels_batch)

                output_batch = output_batch.data.cpu().numpy()
                data_batch = data_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                summary_batch = {metric: metrics[metric](output_batch, labels_batch, 21)
                                for metric in metrics}
                summary_batch['loss'] = loss.item()
                summary_batch['index'] = i
                plot.append(summary_batch['mIOU'])
                summ.append(summary_batch)
                # update the average loss
                loss_avg.update(loss.item())
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()

    sorted_summ = sorted(summ, key=lambda k: k['mIOU'])
    plt.hist(plot, bins=20) 
    plt.title("{} mIOU".format(model_type)) 
    plt.show()
    print(sorted_summ[:5])
    print(sorted_summ[int(len(sorted_summ)/2)-2:int(len(sorted_summ)/2)+2])
    print(sorted_summ[-5:])

    return

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

# Show images
def show_images(data_images, labels_images, output_images, mIOU):
    n=output_images.shape[0]
    for index, (data_image, label_image, output_image) in enumerate(zip(data_images, labels_images, output_images)):
        plt.subplot(n, 3, index*3+1)
        plt.title('input')
        plt.imshow(data_image.transpose(1,2,0))
        plt.axis('off')

        plt.subplot(n, 3, index*3+2)
        plt.title('label')
        plt.imshow(decode_segmap(label_image))
        plt.axis('off')
        #im = Image.fromarray(decode_segmap(label_image))
        #im.save("{}_mask.jpeg".format(mIOU))

        output_image = output_image.transpose(1,2,0)
        output_image = np.argmax(output_image, axis=2)
        plt.subplot(n, 3, index*3+3)
        plt.title('mIOU: {:05.3f}'.format(mIOU))
        plt.imshow(decode_segmap(output_image))
        plt.axis('off')
        #im = Image.fromarray(decode_segmap(output_image))
        #im.save("{}_output.jpeg".format(mIOU))

    plt.show()

# Save images
def save_images(labels_images, output_images, index):
    for _, (label_image, output_image) in enumerate(zip(labels_images, output_images)):
        im = Image.fromarray(decode_segmap(label_image))
        im.save("images\mask\{}.png".format(index))

        output_image = output_image.transpose(1,2,0)
        output_image = np.argmax(output_image, axis=2)
        im = Image.fromarray(decode_segmap(output_image))
        im.save("images\prediction\{}.png".format(index))

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()

    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    params.batch_size = 1

    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    logging.info("Creating the dataset...")

    dataloader = data_loader.fetch_dataloader(['val'], args.data_dir, args.mask_dir, args.dataset_dir, args.num_classes, params)

    test_dl = dataloader['val']

    logging.info("- done.")

    if args.model_type=='GCN_Resnet':
        model = net.GCN_Resnet(args.num_classes).cuda() if params.cuda else net.GCN_Resnet(args.num_classes)
    elif args.model_type=='GCN_Resnet_512':
        model = net.GCN_Resnet_512(args.num_classes).cuda() if params.cuda else net.GCN_Resnet_512(args.num_classes)
    elif args.model_type=='GCN_3':
        model = net.GCN_3(args.num_classes).cuda() if params.cuda else net.GCN_3(args.num_classes)
    elif args.model_type=='GCN_4':
        model = net.GCN_4(args.num_classes).cuda() if params.cuda else net.GCN_4(args.num_classes)
    elif args.model_type=='GCN_3_4':
        model = net.GCN_3_4(args.num_classes).cuda() if params.cuda else net.GCN_3_4(args.num_classes)
    elif args.model_type=='GCN_3_4_C':
        model = net.GCN_3_4_C(args.num_classes).cuda() if params.cuda else net.GCN_3_4_C(args.num_classes)
    elif args.model_type=='GCN_3_4_Linear':
        model = net.GCN_3_4_Linear(args.num_classes).cuda() if params.cuda else net.GCN_3_4_Linear(args.num_classes)
    elif args.model_type=='GCN_3_4_Linear_alpha':
        model = net.GCN_3_4_Linear_alpha(args.num_classes).cuda() if params.cuda else net.GCN_3_4_Linear_alpha(args.num_classes)
    elif args.model_type=='GCN_3_4_alpha':
        model = net.GCN_3_4_alpha(args.num_classes).cuda() if params.cuda else net.GCN_3_4_alpha(args.num_classes)
    else:
        model = net.GCN_Resnet(args.num_classes).cuda() if params.cuda else net.GCN_Resnet(args.num_classes)

    logging.info("- Model Type: {}".format(args.model_type))

    loss_fns = net.loss_fns
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.model_type + '_' + args.restore_file + '.pth.tar'), model)

    # Evaluate
    if args.show_images == 'no':
        evaluate_save_images(model, loss_fns, test_dl, params)
    else:
        evaluate_save_images(model, loss_fns, test_dl, params)

    logging.info("- done.")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if ("alpha" in name) or ("beta" in name) or ("gamma" in name):
                print(name, param.data)