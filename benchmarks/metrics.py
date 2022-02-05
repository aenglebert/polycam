import torch
import torchvision
import math


def blur_image(input_image):
    """
    blur the input image tensor
    """
    return torchvision.transforms.functional.gaussian_blur(input_image, kernel_size=[11, 11], sigma=[5,5])

def grey_image(input_image):
    """
    generate a grey image tensor with same shape as input
    """
    # create an uniform grey image with same size as input_image
    grey = 0.5 * torch.ones_like(input_image).to(input_image.device)
    return grey


class RelevanceMetric:
    """
    base class for Insertion and Deletion
    """
    def __init__(self, model, n_steps, batch_size, baseline="blur"):
        self.model = model
        self.n_steps = n_steps
        self.batch_size = batch_size
        if baseline == "blur":
            self.baseline_fn = blur_image
        elif baseline == "grey":
            self.baseline_fn = grey_image
        else:
            self.baseline_fn = blur_image

    def __call__(self, image, saliency_map, class_idx=None, *args, **kwargs):
        assert image.shape[-2:] == saliency_map.shape[-2:], "Image and saliency map should have the same resolution"

        with torch.no_grad():
            if class_idx is None:
                class_idx = self.model(image).max(1)[1].item()
                print("class not specified, assuming using class ", class_idx)
            h, w = image.shape[-2:]

            # generate baseline
            baseline = self.baseline_fn(image)

            # index of pixels in the saliency map in descending order
            sorted_index = torch.flip(saliency_map.view(-1, h * w).argsort(), dims=[-1])

            # number of batches for this image
            n_batches = (self.n_steps + self.batch_size - 1) // self.batch_size

            # Number of pixels to add or remove per step (could be less for last step)
            pixels_per_steps = (h * w + self.n_steps - 1) // self.n_steps
            pixels_per_batch = pixels_per_steps * self.batch_size

            samples = self.generate_samples(sorted_index, image, baseline)
            # running sum of the scores
            scores = torch.zeros(self.n_steps).to(image.device)

            for idx in range(math.ceil(samples.shape[0] / self.batch_size)):
                selection_slice = slice(idx * self.batch_size, min((idx + 1) * self.batch_size, samples.shape[0]))
                with torch.no_grad():
                    # forward pass, value stored using hook
                    res = self.model(samples[selection_slice])
                    scores[selection_slice] = res[:, class_idx]

            auc = torch.sum(scores) / self.n_steps
            return auc, scores

    def generate_samples(self, *args, **kwargs):
        raise NotImplementedError


class Insertion(RelevanceMetric):
    def __init__(self, model, step, batch_size, baseline="blur"):
        super(Insertion, self).__init__(model, step, batch_size, baseline=baseline)

    def generate_samples(self, index, image, baseline):
        h, w = image.shape[-2:]
        pixels_per_steps = (h * w + self.n_steps - 1) // self.n_steps
        samples = torch.ones(self.n_steps, *image.shape[-3:]).to(image.device)
        samples = samples * baseline
        for step in range(self.n_steps):
            pixels = index[:, :pixels_per_steps*(step+1)]
            samples[step].view(-1, h * w)[..., pixels] = image.view(-1, h * w)[..., pixels]
        return samples


class Deletion(RelevanceMetric):
    def __init__(self, model, step, batch_size, baseline="blur"):
        super(Deletion, self).__init__(model, step, batch_size, baseline=baseline)

    def generate_samples(self, index, image, baseline):
        h, w = image.shape[-2:]
        pixels_per_steps = (h * w + self.n_steps - 1) // self.n_steps
        samples = torch.ones(self.n_steps, *image.shape[-3:]).to(image.device)
        samples = samples * image
        for step in range(self.n_steps):
            pixels = index[:, :pixels_per_steps*(step+1)]
            samples[step].view(-1, h * w)[..., pixels] = baseline.view(-1, h * w)[..., pixels]
        return samples


class AverageDropIncrease:
    """
    Return a tuple of [AverageDrop, IncreaseOfConfidence]
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, image, saliency_map, class_idx=None, *args, **kwargs):
        assert image.shape[-2:] == saliency_map.shape[-2:], "Image and saliency map should have the same resolution"

        with torch.no_grad():
            if class_idx is None:
                class_idx = self.model(image).max(1)[1].item()
                #print("class not specified, assuming using class ", class_idx)
            h, w = image.shape[-2:]

        mask = saliency_map - saliency_map.min()/ saliency_map.max() - saliency_map.min()
        masked_image = mask * image

        base_score = torch.softmax(self.model(image), dim=1)[:, class_idx]
        score = torch.softmax(self.model(masked_image), dim=1)[:, class_idx]
        drop = torch.maximum(torch.zeros(1).to(base_score.device), base_score - score) / base_score
        increase = int(score > base_score)
        return drop.item(), increase
