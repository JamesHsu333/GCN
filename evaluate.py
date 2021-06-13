import argparse
import itertools 
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt
import model.net as net
import model.data_loader as data_loader

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
parser.add_argument('--num_classes', default=21,
                    help="Numbers of classes")

def evaluate(model, loss_fns, dataloader, evaluator, params):
    model.eval()
    #model.apply(net.deactivate_batchnorm)
    evaluator.reset()

    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for data_batch, labels_batch in dataloader:
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

                evaluator.add_batch(labels_batch, output_batch)
                # update the average loss
                loss_avg.update(loss.item())
                
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()
    metrics_mean = {'mIOU': evaluator.Mean_Intersection_over_Union(), 'loss': loss_avg()}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()

    params.batch_size = 1

    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    logging.info("Creating the dataset...")

    dataloader = data_loader.fetch_dataloader(['val'], args.data_dir, args.mask_dir, args.dataset_dir, args.num_classes, params)

    test_dl = dataloader['val']

    logging.info("- done.")

    if args.model_type=='GCN_Resnet':
        model = net.GCN_Resnet(args.num_classes).cuda() if params.cuda else net.GCN_Resnet(args.num_classes)
    elif args.model_type=='GCN_Resnet_512':
        model = net.GCN_Resnet_512(args.num_classes).cuda() if params.cuda else net.GCN_Resnet_512(args.num_classes)
    elif args.model_type=='GCN_1':
        model = net.GCN_1(args.num_classes).cuda() if params.cuda else net.GCN_1(args.num_classes)
    elif args.model_type=='GCN_1_L':
        model = net.GCN_1_L(args.num_classes).cuda() if params.cuda else net.GCN_1_L(args.num_classes)
    elif args.model_type=='GCN_2':
        model = net.GCN_2(args.num_classes).cuda() if params.cuda else net.GCN_2(args.num_classes)
    elif args.model_type=='GCN_2_L':
        model = net.GCN_2_L(args.num_classes).cuda() if params.cuda else net.GCN_2_L(args.num_classes)
    elif args.model_type=='GCN_3':
        model = net.GCN_3(args.num_classes).cuda() if params.cuda else net.GCN_3(args.num_classes)
    elif args.model_type=='GCN_3_L':
        model = net.GCN_3_L(args.num_classes).cuda() if params.cuda else net.GCN_3_L(args.num_classes)
    elif args.model_type=='GCN_4':
        model = net.GCN_4(args.num_classes).cuda() if params.cuda else net.GCN_4(args.num_classes)
    elif args.model_type=='GCN_3_4':
        model = net.GCN_3_4(args.num_classes).cuda() if params.cuda else net.GCN_3_4(args.num_classes)
    elif args.model_type=='GCN_3_4_C':
        model = net.GCN_3_4_C(args.num_classes).cuda() if params.cuda else net.GCN_3_4_C(args.num_classes)
    elif args.model_type=='GCN_3_times_4':
        model = net.GCN_3_times_4(args.num_classes).cuda() if params.cuda else net.GCN_3_times_4(args.num_classes)
    elif args.model_type=='GCN_3_4_L':
        model = net.GCN_3_4_L(args.num_classes).cuda() if params.cuda else net.GCN_3_4_L(args.num_classes)
    elif args.model_type=='GCN_3_times_4_L':
        model = net.GCN_3_times_4_L(args.num_classes).cuda() if params.cuda else net.GCN_3_times_4_L(args.num_classes)
    elif args.model_type=='GCN_3_times_4_NoSigmoid':
        model = net.GCN_3_times_4_NoSigmoid(args.num_classes).cuda() if params.cuda else net.GCN_3_times_4_NoSigmoid(args.num_classes)
    elif args.model_type=='GCN_3_times_4_NoSigmoid_L':
        model = net.GCN_3_times_4_NoSigmoid_L(args.num_classes).cuda() if params.cuda else net.GCN_3_times_4_NoSigmoid_L(args.num_classes)
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
    evaluator = net.Evaluator(20+1)

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.model_type + '_' + args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fns, test_dl, evaluator, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
    logging.info("- done.")

    for name, param in model.named_parameters():
        if param.requires_grad:
            if ("alpha" in name) or ("beta" in name) or ("gamma" in name):
                print(name, torch.nn.Softmax(param.data))