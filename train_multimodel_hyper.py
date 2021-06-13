import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/GCN_Resnet',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='../../Dataset/VOC2012/JPEGImages',
                    help="Directory containing the dataset")
parser.add_argument('--mask_dir', default='../../Dataset/VOC2012/SegmentationClass',
                    help="Directory containing the mask dataset")


def launch_training_job(parent_dir, data_dir, mask_dir, model_type, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir} --mask_dir {mask_dir} --model_type {model_type}".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir, mask_dir=mask_dir, model_type=model_type)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    learning_rates = [1e-4, 5e-5]

    models = ['GCN_Resnet', 'GCN_3', 'GCN_4', 'GCN_3_4', 'GCN_3_4_L', 'GCN_3_4_Linear', 'GCN_3_4_alpha', 'GCN_3_4_Linear_alpha']
    
    for model in models:
        for learning_rate in learning_rates:
            # Modify the relevant parameter in params
            params.learning_rate = learning_rate

            # Launch job (name has to be unique)
            job_name = "learning_rate_{}".format(learning_rate)
            launch_training_job(args.parent_dir, args.data_dir, args.mask_dir, model, job_name, params)