import argparse

import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

from benchmarks.imagesDataset import ImagesDataset
from benchmarks.utils import imagenet_labels, overlay
from benchmarks.metrics import Insertion, Deletion

import numpy as np

import pandas as pd

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Saliency map faithfullness metrics evaluation from npz")

#########################
#### data parameters ####
#########################
parser.add_argument("--image_folder", type=str, default='images',
                    help="path to images repository")

parser.add_argument("--image_list", type=str, default='images.txt',
                    help="path to images list file")

parser.add_argument("--labels_list", type=str, default='imagenet_validation_imagename_labels.txt',
                    help="path to labels list file in txt")

parser.add_argument("--model", type=str, default='vgg16',
                    help="model type: vgg16 (default) or resnet50")

parser.add_argument("--saliency_npz", type=str, default='',
                    help="saliency file")

parser.add_argument("--cuda", dest="gpu", action='store_true',
                    help="use cuda")
parser.add_argument("--cpu", dest="gpu", action='store_false',
                    help="use cpu instead of cuda (default)")
parser.set_defaults(gpu=False)


parser.add_argument("--npz_folder", type=str, default="./npz",
                    help="Path to the folder where npz are stored")
parser.add_argument("--csv_folder", type=str, default="./csv",
                    help="Path to the folder to store the csv outputs")
parser.add_argument("--batch_size", type=int, default=1,
                    help="max batch size, default to 1")

def main():
    global args
    args = parser.parse_args()

    # Model selection
    if args.model == 'resnet50':
        model = models.resnet50(True)
    elif args.model == 'vgg16':
        model = models.vgg16(True)
    else:
        print("model: " + args.model + " unknown, set to resnet18 by default")
        model = models.resnet18(True)

    model.eval()
    model_softmax = torch.nn.Sequential(model, torch.nn.Softmax(dim=-1))
    if args.gpu:
        model = model.cuda()
        model_softmax.cuda()

    input_size = 224

    # get saliencies from file
    saliencies = np.load(args.npz_folder + "/" + args.saliency_npz)

    # Set metrics, use input size as step size
    insertion = Insertion(model_softmax, input_size, args.batch_size)
    deletion = Deletion(model_softmax, input_size, args.batch_size)

    # Set tranform (use ImageNet trainset mean and sd)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((input_size, input_size)),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    # Initiate dataset
    dataset = ImagesDataset(args.image_list, args.labels_list, args.image_folder, transform=transform)

    ins_auc_dict = dict()
    del_auc_dict = dict()
    ins_details_dict = dict()
    del_details_dict = dict()

    for i in tqdm(range(len(dataset))):
        # import image from dataset
        image, label, image_name = dataset[i]
        image = image.unsqueeze(0)
        saliency = torch.tensor(saliencies[image_name])

        # upscale saliency
        sh, sw = saliency.shape[-2:]
        saliency = saliency.view(1,1,sh,sw)
        saliency = F.interpolate(saliency, image.shape[-2:], mode='bilinear')

        # set image and saliency to gpu if required
        if args.gpu:
            image = image.cuda()
            saliency = saliency.cuda()

        # get class predicted by the model for the full image, it's the class used to generate saliency map
        class_idx = model(image).max(1)[1].item()

        # compute insertion and deletion for each step + auc on the image
        ins_auc, ins_details = insertion(image, saliency, class_idx=class_idx)
        del_auc, del_details = deletion(image, saliency, class_idx=class_idx)

        # store every values for the image in dictionary
        ins_auc_dict[image_name] = ins_auc.cpu().numpy()
        ins_details_dict[image_name] = ins_details.cpu().numpy()
        del_auc_dict[image_name] = del_auc.cpu().numpy()
        del_details_dict[image_name] = del_details.cpu().numpy()


    csv_suffix = '.'.join(args.saliency_npz.split('.')[:-1]) + ".csv"

    # save in csv files
    pd.DataFrame.from_dict(ins_auc_dict, orient='index').to_csv(args.csv_folder + "/" + 'ins_auc_' + csv_suffix)
    pd.DataFrame.from_dict(del_auc_dict, orient='index').to_csv(args.csv_folder + "/" + 'del_auc_' + csv_suffix)
    pd.DataFrame.from_dict(ins_details_dict, orient='index').to_csv(args.csv_folder + "/" + 'ins_details_' + csv_suffix)
    pd.DataFrame.from_dict(del_details_dict, orient='index').to_csv(args.csv_folder + "/" + 'del_details_' + csv_suffix)

if __name__ == "__main__":
    main()
