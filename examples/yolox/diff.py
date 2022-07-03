#!/usr/bin/env python3

import numpy as np

import torch

import invis as inv

np.random.seed(3)


def build_torch_yolox():
    from yolox import YOLOX
    from yolo_pafpn import YOLOPAFPN
    from yolo_head import YOLOXHead
    in_channels = [256, 512, 1024]
    depth, width = 0.33, 0.50
    act = "silu"
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head = YOLOXHead(80, width, in_channels=in_channels, act=act)
    model = YOLOX(backbone, head)
    return model


def build_invis_yolox():
    from yolox_invis import YOLOX
    from yolo_pafpn_invis import YOLOPAFPN
    from yolo_head_invis import YOLOXHead
    in_channels = [256, 512, 1024]
    depth, width = 0.33, 0.50
    act = "silu"
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head = YOLOXHead(80, width, in_channels=in_channels, act=act)
    model = YOLOX(backbone, head)
    return model


def check_output(
    mge_module, torch_module, input_shape=(1, 3, 640, 640), repeat=5, rtol=1e-05, atol=1e-08
):
    for _ in range(repeat):
        inputs1 = np.random.normal(size=input_shape)
        inputs2 = inputs1.copy()
        mge_output = mge_module(inv.Tensor(inputs2)).numpy()
        torch_output = torch_module(torch.Tensor(inputs1)).detach().cpu().numpy()
        if not np.allclose(torch_output, mge_output, rtol, atol):
            return False
    return True


def main():
    torch_model = build_torch_yolox()
    weights = torch.load("yolox_s.pth", map_location="cpu")["model"]
    torch_model.load_state_dict(weights)
    torch_model.eval()

    mge_model = build_invis_yolox()
    mge_model.load_state_dict(weights)
    mge_model.eval()
    print("check value: ", check_output(mge_model, torch_model, rtol=1e-2, atol=1e-4))


if __name__ == "__main__":
    main()
