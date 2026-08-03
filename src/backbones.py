"""
Frozen backbones -> feature vectors.

MVP trio, each a ViT-B loadable in one line:
  mae         reconstructive SSL      timm 'vit_base_patch16_224.mae'
  dinov2      joint-embedding SSL     torch.hub facebookresearch/dinov2 'dinov2_vitb14'
  supervised  label-trained control   timm 'vit_base_patch16_224.augreg_in21k_ft_in1k'

Everything is frozen (eval + no_grad). We take one pooled embedding per image, so the
downstream step is just a linear probe. This is the whole reason the MVP is cheap: no
backbone is ever trained.

FULL STUDY: load_ijepa() sketches loading a true I-JEPA checkpoint from the official repo
(github.com/facebookresearch/ijepa). It is untested here (needs the checkpoint + their
key layout), so verify the state_dict remap against your file before trusting numbers.
"""
from __future__ import annotations
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _timm_model_and_tf(name, img_size):
    import timm
    from timm.data import resolve_data_config, create_transform
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.eval()
    cfg = resolve_data_config({"input_size": (3, img_size, img_size)}, model=model)
    tf = create_transform(**cfg)  # PIL -> normalized tensor
    return model, tf


def _basic_tf(img_size):
    from torchvision import transforms as T
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_backbone(name, img_size=224, device="cpu"):
    """Return (model, transform, embed_fn). embed_fn(model, batch_tensor)->[B,D] numpy."""
    name = name.lower()
    if name == "mae":
        model, tf = _timm_model_and_tf("vit_base_patch16_224.mae", img_size)
    elif name == "supervised":
        model, tf = _timm_model_and_tf("vit_base_patch16_224.augreg_in21k_ft_in1k", img_size)
    elif name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model.eval()
        tf = _basic_tf(img_size)
    elif name == "ijepa":
        model, tf = load_ijepa(img_size=img_size)
    else:
        raise KeyError(f"unknown backbone {name!r}")
    model = model.to(device)

    @torch.no_grad()
    def embed(m, x):
        out = m(x.to(device))
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 3:            # [B, tokens, D] -> mean pool
            out = out.mean(dim=1)
        return out.float().cpu().numpy()

    return model, tf, embed


def load_ijepa(ckpt_path=None, arch="vit_base_patch16_224", img_size=224):
    """FULL-STUDY sketch. Downloads/loads an I-JEPA target-encoder checkpoint into a timm
    ViT of matching arch. NOTE: I-JEPA's released configs are often ViT-H/14, not B/16 --
    match `arch` to your checkpoint. Untested here; treat as a starting point."""
    import timm
    from timm.data import resolve_data_config, create_transform
    if ckpt_path is None:
        raise NotImplementedError(
            "Set ckpt_path to a local I-JEPA checkpoint from "
            "https://github.com/facebookresearch/ijepa (see their README for URLs), "
            "and match `arch` to the checkpoint (e.g. vit_huge_patch14_224)."
        )
    model = timm.create_model(arch, pretrained=False, num_classes=0)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("target_encoder", sd.get("encoder", sd))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ijepa] loaded with {len(missing)} missing / {len(unexpected)} unexpected keys")
    model.eval()
    cfg = resolve_data_config({"input_size": (3, img_size, img_size)}, model=model)
    return model, create_transform(**cfg)
