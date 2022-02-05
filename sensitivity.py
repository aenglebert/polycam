import argparse

import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

from benchmarks.imagesDataset import ImagesDataset
from benchmarks.utils import ZoomCAMResnet

from polycam.polycam import  PCAMp, PCAMm, PCAMpm

try:
    from torchcam.cams import GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISCAM, LayerCAM
    torchcam = True
except:
    print("torchcam not installed: GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISCAM not availables")
    torchcam = False

try:
    from RISE.explanations import RISE
    rise = True
except:
    print("RISE not installed, not available")
    rise = False

try:
    from captum.attr import IntegratedGradients, InputXGradient, Lime, Occlusion, Saliency, NoiseTunnel
    captum = True
except:
    print("captum not installed, IntegratedGradients, InputXGradient, SmoothGrad, Occlusion not availables")
    captum = False

try:
    from zoomcam import object_class
    from zoomcam import function
    zoomcam = True
except:
    print("zoomcam not found")
    zoomcam = False

import numpy as np

import pandas as pd

from captum.metrics import sensitivity_max

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Saliency map methods evaluation")

#########################
#### data parameters ####
#########################
parser.add_argument("--image_folder", type=str, default='images',
                    help="path to images repository")

parser.add_argument("--image_list", type=str, default='images.txt',
                    help="path to images list file")
parser.add_argument("--labels_list", type=str, default='imagenet_validation_imagename_labels.txt',
                    help="path to labels list file")

parser.add_argument("--model", type=str, default='vgg16',
                    help="model type: vgg16 (default) or resnet50")

parser.add_argument("--saliency", type=str, default='pcampm',
                    help="saliency type: pcamp, pcamm, pcampm (default), gradcam, gradcampp, smoothgradcampp, "
                         "ig (=IntegratedGrad), ixg (=InputxGrad), sg (=SmoothGrad), occlusion, rise")

parser.add_argument("--cuda", dest="gpu", action='store_true',
                    help="use cuda")
parser.add_argument("--cpu", dest="gpu", action='store_false',
                    help="use cpu instead of cuda (default)")
parser.set_defaults(gpu=False)

parser.add_argument("--batch_size", type=int, default=1,
                    help="max batch size (when saliency method use it), default to 1")

parser.add_argument("--csv_folder", type=str, default="./csv",
                    help="Path to the folder to store the output file")
parser.add_argument("--suffix", type=str, default="",
                    help="Add SUFFIX string to the output file name")



def pcam_wrapper_fn(pcam):
    def fn(input, target):
        output = pcam(input[0], class_idx=target)[-1]
        return output
    return fn

def overlay_wapper_fn(overlay):
    def fn(input, target):
        torch.set_grad_enabled(True)
        input_grad = input[0].clone().detach().requires_grad_(True)
        output = overlay(input_grad, class_idx=target)
        return output
    return fn

def rise_wrapper_fn(rise):
    def fn(input, target):
        return rise(input[0])[target]
    return fn

def torchcam_wrapper_fn(torchcam, model):
    def fn(input, target):
        torch.set_grad_enabled(True)
        input_grad = input[0].clone().detach().requires_grad_(True)
        out = model(input_grad)
        saliency_map = torchcam(target, out)
        saliency_map = saliency_map.view(1, 1, *saliency_map.shape)
        return saliency_map
    return fn

def main():
    global args
    args = parser.parse_args()

    # Model selection
    target_layer = None
    if args.model == 'resnet50':
        model = models.resnet50(True)
        target_layers = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
    elif args.model == 'vgg16':
        model = models.vgg16(True)
        # target layers for cam methods
        target_layer = "features.29"
        target_layers = ['features.3', 'features.8', 'features.15', 'features.22', 'features.29']
    else:
        print("model: " + args.model + " unknown, set to vgg16 by default")
        model = models.vgg16(True)

    if args.saliency.lower() == 'rise':
        if not rise:
            print("Cannot use rise, import not available")
            return
        model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
        for p in model.parameters():
            p.requires_grad = False

    if args.saliency.lower() in ["ig", "ixg", "occlusion", "lime", "sg"] and not captum:
        print("cannot use captum methods, import not available")
        return

    if args.saliency.lower() in ["gradcam", "scorecam", "gradcampp", "smoothgradcampp", "sscam", "iscam"] and not torchcam:
        print("cannot use torchcam methods, import not available")
        return

    model.eval()
    # set model to cuda if required
    if args.gpu:
        model = model.cuda()

    input_size = 224

    # Saliency selection
    n_maps = 1
    library = None
    if args.saliency.lower() == 'pcamp':
        saliency = PCAMp(model, batch_size=args.batch_size)
        n_maps = 5
        library = "polycam"
    elif args.saliency.lower() == 'pcamm':
        saliency = PCAMm(model, batch_size=args.batch_size)
        n_maps = 5
        library = "polycam"
    elif args.saliency.lower() == 'pcampm':
        saliency = PCAMpm(model, batch_size=args.batch_size)
        n_maps = 5
        library = "polycam"
    elif args.saliency.lower() == 'gradcam':
        saliency = GradCAM(model, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'gradcampp':
        saliency = GradCAMpp(model, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'smoothgradcampp':
        saliency = SmoothGradCAMpp(model, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'scorecam':
        saliency = ScoreCAM(model, batch_size=args.batch_size, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'sscam':
        saliency = SSCAM(model, batch_size=args.batch_size, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'iscam':
        saliency = ISCAM(model, batch_size=args.batch_size, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower()[:-1] == 'layercamfusion':
        n_layers = int(args.saliency.lower()[-1])
        first_layer_idx = len(target_layers) - n_layers - 1
        print(first_layer_idx)
        layercam_dict = {}
        for layer in target_layers[first_layer_idx:]:
            layercam_dict[layer] = LayerCAM(model, target_layer=layer)
        def scale_fn(cam, factor):
            return torch.tanh((factor * cam) / cam.max())
        def layer_fusion(inputs, class_idx=0):
            out = model(inputs)
            cam_fusion = torch.zeros((1,1,1,1))
            if args.gpu:
                cam_fusion = cam_fusion.cuda()
            for layer in reversed(target_layers[first_layer_idx:]):
                saliency_map = layercam_dict[layer](class_idx, out)
                saliency_map = saliency_map.view((1,1,saliency_map.shape[-2],saliency_map.shape[-1]))
                if layer in (target_layers[:3]):
                    saliency_map = scale_fn(saliency_map, 2)
                cam_fusion = F.interpolate(cam_fusion, saliency_map.shape[-2:], mode="bilinear")
                cam_fusion = torch.maximum(cam_fusion, saliency_map)
            return cam_fusion   
        saliency = layer_fusion
        library = "overlay"
    elif args.saliency.lower() == 'zoomcam':
        if args.model == "vgg16":
            zoom_layers = ['1', '3', '6', '8', '11', '13', '15', '18', '20', '22', '25', '27', '29']
        elif args.model == "resnet50":
            zoom_layers = target_layers
            model = ZoomCAMResnet(model)
        else:
            print("model not available for this method")
            return



        def zoomcam_fn(inputs, class_idx=0):
            device = inputs.device

            zoom_cam_model = object_class.ZoomCAM_gradients(model, zoom_layers)

            activations, weights = zoom_cam_model(inputs, device, class_idx)

            for i in range(len(zoom_layers)):
                zoom_cam = activations[len(zoom_layers) - 1 - i].cpu().detach() * weights[i].cpu().detach()
                zoom_cam = function.normalize_CAM(torch.sum(zoom_cam, dim=(0, 1), keepdim=True))
                if i == 0:
                    aggregated_zoom_cam = function.normalize_CAM(torch.sum(zoom_cam, dim=(0, 1), keepdim=True))
                else:
                    scale = zoom_cam.size(2) / aggregated_zoom_cam.size(2)
                    upsample = torch.nn.Upsample(scale_factor=scale, mode='bilinear')
                    aggregated_zoom_cam = torch.max(upsample(aggregated_zoom_cam), zoom_cam)

            # delete to clear memory
            del activations
            del weights
            del zoom_cam_model.extractor.weights
            del zoom_cam_model.extractor.activations
            del zoom_cam_model.extractor
            del zoom_cam_model
            del zoom_cam
            return aggregated_zoom_cam.detach()

        saliency = zoomcam_fn
        library = "overlay"
    elif args.saliency.lower() == 'rise':
        saliency = RISE(model, (input_size, input_size), args.batch_size)
        saliency.generate_masks(N=6000, s=8, p1=0.1)
        library = "rise"
    elif args.saliency.lower() == 'ig':
        saliency = IntegratedGradients(model)
        library = "captum"
    elif args.saliency.lower() == 'ixg':
        saliency = InputXGradient(model)
        library = "captum"
    elif args.saliency.lower() == 'sg':
        gradient = Saliency(model)
        sg = NoiseTunnel(gradient)
        def sg_fn(inputs, class_idx=0):
            return sg.attribute(inputs, nt_samples=50, nt_samples_batch_size=args.batch_size, target=class_idx).sum(1)
        saliency = sg_fn
        library = "overlay"
    elif args.saliency.lower() == 'occlusion':
        occlusion = Occlusion(model)
        def occ_fn(inputs, class_idx=0):
            return occlusion.attribute(inputs, target=class_idx, sliding_window_shapes=(3,64,64), strides=(3,8,8)).sum(1)
        saliency = occ_fn
        library = "overlay"


    # wrapper function to enter into the captum sensitivity function

    if library == 'polycam':
        attribute_fn = pcam_wrapper_fn(saliency)
    elif library == 'captum':
        attribute_fn = saliency.attribute
    elif library == 'rise':
        attribute_fn = rise_wrapper_fn(saliency)
    elif library == "torchcam":
        attribute_fn = torchcam_wrapper_fn(saliency, model)
    elif library == 'overlay':
        attribute_fn = overlay_wapper_fn(saliency)

    # Set tranforms, normalise to ImageNet train mean and sd
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((input_size, input_size)),
                                    transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    # Initiate dataset
    dataset = ImagesDataset(args.image_list, args.labels_list, args.image_folder, transform=transform)


    # Handle multiple saliency maps if needed
    if n_maps > 1:
        saliencies = []
        for n in range(n_maps):
            saliencies.append(dict())
    else:
        saliencies = dict()

    # loop over all the dataset. Can be changed to process only part of the dataset
    range_from = 0
    range_to = len(dataset)

    sensitivity_dict = dict()

    # loop over the dataset
    for i in tqdm(range(range_from, range_to), desc='generating saliency maps'):
        sample, labels, sample_name = dataset[i]
        sample = sample.unsqueeze(0)
        if args.gpu:
            sample = sample.cuda()
        out = model(sample)
        class_idx = out.squeeze(0).argmax().item()
        # sensitivity check for selected method
        sens = sensitivity_max(attribute_fn, sample, target=class_idx, max_examples_per_batch=1)
        sensitivity_dict[sample_name] = sens.cpu().item()

    csv_name = "sens_" + args.model + "_" + args.saliency + args.suffix + ".csv"

    # save in csv files
    pd.DataFrame.from_dict(sensitivity_dict, orient='index').to_csv(args.csv_folder + "/" + csv_name)

if __name__ == "__main__":
    main()
