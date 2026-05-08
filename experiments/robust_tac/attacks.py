import torch

from losses import kl_target_to_prediction


def pgd_cluster_attack(encoder, cluster_head, images, p_clean, eps=1.0 / 255.0, steps=1, step_size=None):
    if steps <= 0 or eps <= 0:
        return images.detach()

    step_size = eps if step_size is None else step_size
    delta = torch.zeros_like(images).uniform_(-eps, eps)
    delta = torch.clamp(images + delta, 0.0, 1.0) - images

    for _ in range(steps):
        delta.requires_grad_(True)
        adv_images = torch.clamp(images + delta, 0.0, 1.0)
        adv_features = encoder(adv_images)
        p_adv = cluster_head.image_forward(adv_features)
        loss = kl_target_to_prediction(p_clean.detach(), p_adv)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        delta = delta.detach() + step_size * torch.sign(grad.detach())
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(images + delta, 0.0, 1.0) - images

    return torch.clamp(images + delta.detach(), 0.0, 1.0)
