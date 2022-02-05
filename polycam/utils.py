import torch
import torch.nn.functional as F


def find_layer_list(model, input_shape=(3,224,224)):
    """
    find the target layers for deepcam
    inspired by the implementation from https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/utils.py
    :param model: the CNN to analyse
    :param input_shape: shape of the input image (c,h,w), default to imagenet like 3,224,224
    :return: list of target layers' names
    """

    # get the model device
    device = next(model.parameters()).device

    # set to eval mode
    model.eval()

    # list to keep track of layers shapes
    layer_shapes = []

    # Define hook function to append module's names and output shapes to the list
    def set_layer_hook(name):
        def fn(module, input, output):
            layer_shapes.append((name, output.shape))
        return fn

    # keep track of hooks for easier removal
    hooks_list = []

    # set hooks to every module
    for name, module in model.named_modules():
        hooks_list.append(module.register_forward_hook(set_layer_hook(name)))

    # forward pass with dummy input
    dummy_input = torch.rand(1, *input_shape).to(device)
    with torch.no_grad():
        model(dummy_input)

    # remove the hooks
    for hook in hooks_list:
        hook.remove()

    target_list = []
    # skip the first layer, mostly needed if the stride of the first layer is > 1
    last_layer_name = layer_shapes[0][0]
    last_output_shape = layer_shapes[0][1]

    # iterate through the layers to find target layers
    for layer_name, output_shape in layer_shapes[1:]:
        # target layers are layer before the changing in shape for the two last dimensions
        if output_shape[2:] != last_output_shape[2:]:
            # don't save and break if flatten if encountered
            if not len(output_shape) != len(last_output_shape):
                target_list.append(last_layer_name)
            else:
                break
            # stop after saving the last layer if the last two dimensions are (1,1)
            if all((output_shape[-1] == 1, output_shape[-2] == 1)):
                break
        # keep track of the previous name and shape for next iteration
        last_output_shape = output_shape
        last_layer_name = layer_name

    return target_list


def softmax_sigmoid(input, dim=None):
    """
    function that return a softmax, except if the input dimension only have a shape of 1, then return a sigmoid
    :param input: input tensor
    :param dim: dimension to compute the softmax
    :return: softmax or sigmoid of the input
    """
    if dim is not None and input.shape[dim] > 1:
        return F.softmax(input, dim=dim)
    elif not all([v == 1 for v in input.shape]):
        return F.softmax(input)
    return F.sigmoid(input)
