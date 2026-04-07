import torch
import cv2
import numpy as np

from config.config import IMG_SIZE


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()

        output = self.model(input_tensor)

        # Focus on strongest activation region
        loss = output.max()

        loss.backward()

        grads = self.gradients[0]       # [C, H, W]
        acts = self.activations[0]      # [C, H, W]

        weights = torch.mean(grads, dim=(1, 2))

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32).to(acts.device)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam)

        if torch.max(cam) != 0:
            cam = cam / torch.max(cam)

        cam = cam.cpu().detach().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        return cam  

