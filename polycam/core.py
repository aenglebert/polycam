import torch
import torch.nn.functional as F
from .utils import *
import numpy as np


class _PolyCAM:
    """
    Base class for Poly Class Activation Mapping
    :param model: the CNN to analyse
    :param target_layer_list: list of the target layers, automatic discovery if not provided, optional
    :param intermediate_maps: return intermediate maps without local normalisation, not polycam (for ablation study)
    :param lnorm: default to True, usage of LNorm (for ablation study)
    """
    def __init__(self, model, target_layer_list=None, intermediate_maps=False, lnorm=True):
        self.model = model
        # find target layers if not provided
        if target_layer_list is None:
            self.target_layer_list = find_layer_list(model)
        else:
            self.target_layer_list = target_layer_list

        # dict to store activations of target layers
        self.target_activations = dict()

        # hooks list
        self.hooks = []

        ## Ablation study parameters
        # bool to return intermediate cam instead of polycam
        self.intemediate_maps = intermediate_maps

        # usage of LNorm
        self.lnorm = lnorm

    def set_activation_hooks(self):
        # define a hook that store activations in the dict with module's name as key
        def activation_hook(name):
            def fn(_, __, output):
                self.target_activations[name] = output
            return fn

        # set hooks to target layers
        for layer in self.target_layer_list:
            self.hooks.append(dict(self.model.named_modules())[layer].register_forward_hook(activation_hook(layer)))

    def unset_activation_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def _pre_call(self, input_image, class_idx):
        # enable activations hooks
        self.set_activation_hooks()

        # first pass to retrieve baseline activations of target layers
        result = self.model(input_image)

        # disable activations hooks
        self.unset_activation_hooks()

        # get class_idx from result if not provided
        if class_idx is None:
            self.class_idx = result.max(1)[-1].item()
        else:
            self.class_idx = class_idx
        # keep baseline score
        baseline_score = softmax_sigmoid(result)[:, class_idx].view(-1, 1, 1)

        self.score = baseline_score

    def __call__(self, input_image, class_idx=None):
        self._pre_call(input_image, class_idx)

        # find last layer to assign first score hook
        last_layer = None
        for name, module in self.model.named_modules():
            last_layer = module

        # scores are computed relatively to the output of the softmax
        def _last_layer_score_hook():
            def fn(_, __, output):
                self.score = softmax_sigmoid(output)[:, self.class_idx].view(-1, 1, 1)
            return fn

        score_hook = last_layer.register_forward_hook(_last_layer_score_hook())

        if self.intemediate_maps is False:
            cam_list = self.compute_cams(input_image)
        else:
            cam_list = self.compute_intermediate_cams(input_image)

        # remove score hook
        score_hook.remove()
        return cam_list

    def compute_cams(self, input_image):
        activation_map = None
        prev_cam = None
        cam_list = []

        # compute cam in reverse order on target layers
        for layer in reversed(self.target_layer_list):
            activations = self.target_activations[layer]
            weights = self.get_weights(input_image, layer)
            activation_map = activations * weights
            activation_map = activation_map.nansum(1).unsqueeze(0)
            activation_map = torch.relu(activation_map)
            if prev_cam is None:
                prev_cam = activation_map
            else:
                # multiplex with previously computed CAM from later layer
                if self.lnorm:
                    activation_map = self.local_relevance_normalization(activation_map, prev_cam.shape[-2:])
                prev_cam = activation_map * F.interpolate(prev_cam, activation_map.shape[-2:], mode="bilinear")
            prev_cam = torch.relu(prev_cam)
            cam_list.append(F.interpolate(self.normalize(prev_cam), input_image.shape[-2:], mode="bilinear"))
        return cam_list

    def compute_intermediate_cams(self, input_image):
        activation_map = None
        prev_cam = None
        cam_list = []

        # compute cam in reverse order on target layers
        for layer in reversed(self.target_layer_list):
            activations = self.target_activations[layer]
            weights = self.get_weights(input_image, layer)
            activation_map = activations * weights
            activation_map = activation_map.nansum(1).unsqueeze(0)
            activation_map = torch.relu(activation_map)
            cam_list.append(activation_map)
        return cam_list

    def get_weights(self, baseline_input, layer):
        raise NotImplementedError

    @staticmethod
    def local_relevance_normalization(input_tensor, norm_size):
        """
        Local normalisation, normalise values by groups of (norm_size, norm_size).
        """
        norm_factor = F.interpolate(input_tensor, norm_size, mode="bilinear")
        norm_factor = F.interpolate(norm_factor, input_tensor.shape[-2:], mode="bilinear")
        norm_factor = torch.maximum(norm_factor, torch.tensor(1e-12))
        return input_tensor/norm_factor

    @staticmethod
    def normalize(input_tensor, outlier_perc=0.1):
        """
        Normalise in [0, 1] interval
        """
        # keep device to return tensor in the same device as input
        device = input_tensor.device
        eps = torch.ones(*input_tensor.shape[:-2], 1, 1).to(device) * 1e-12
        # define min and max for normalization in [0,1] range
        min = np.percentile(input_tensor.cpu().detach().flatten(-2), outlier_perc, axis=-1)
        min = torch.tensor(min).view([*input_tensor.shape[:-2], 1, 1]).float().to(device)
        max = np.percentile(input_tensor.cpu().detach().flatten(-2), 100-outlier_perc, axis=-1)
        max = torch.tensor(max).view([*input_tensor.shape[:-2], 1, 1]).float().to(device)
        # normalize from min and max
        normalized = input_tensor - min
        normalized = normalized / torch.maximum(max, eps)
        normalized = torch.minimum(normalized, torch.ones(1).to(device))
        normalized = torch.maximum(normalized, torch.zeros(1).to(device))
        return normalized.to(device)
