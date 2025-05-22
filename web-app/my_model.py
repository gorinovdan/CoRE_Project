from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np

def run_gradcam(model, x, rule_index=0):
    from torch import nn
    class RegressionChannelWrapper(nn.Module):
        def __init__(self, model, channel_index=0):
            super().__init__()
            self.model = model
            self.channel_index = channel_index
        def forward(self, x):
            out = self.model(x)
            return out['regression'][:, self.channel_index].unsqueeze(1)
    wrapped_model = RegressionChannelWrapper(model, channel_index=rule_index)
    target_layer = model.backbone.blocks[-1]  # ViT
    cam = GradCAM(
        model=wrapped_model,
        target_layers=[target_layer],
        reshape_transform=None  # Если нужно, добавь функцию для ViT reshape
    )
    grayscale = cam(input_tensor=x)[0]
    rgb = x[0].permute(1,2,0).detach().cpu().numpy()
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    heatmap = show_cam_on_image(rgb, grayscale, use_rgb=True)
    return heatmap  # np.ndarray

def run_shap(model, x, rule_index=0):
    import shap
    background = x[:16]  # или другой batch из train
    e = shap.DeepExplainer(model, background)
    shap_vals = e.shap_values(x[:1])  # на одно изображение
    # Преобразовать в картинку
    import matplotlib.pyplot as plt
    shap_img = shap.image_plot([shap_vals[rule_index]], x[:1].cpu().numpy(), show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)
