#!/usr/bin/env python3

import numpy as np
import invis as torch
from torch import Tensor
np.random.seed(3)


def check_output(
    mge_module, torch_module, input_shape=(1, 3, 224, 224), repeat=5, rtol=1e-05, atol=1e-08
):
    for _ in range(repeat):
        inputs = np.random.normal(size=input_shape)
        torch_output = torch_module(Tensor(inputs)).detach().cpu().numpy()
        mge_output = mge_module(torch.Tensor(inputs)).numpy()
        if not np.allclose(torch_output, mge_output, rtol, atol):
            return False
    return True


if __name__ == "__main__":
    from alexnet import alexnet
    from alexnet_invis import alexnet as mge_alexnet

    model = alexnet(pretrained=True)
    mge_model = mge_alexnet(pretrained=True)
    model.eval()
    mge_model.eval()

    print(
        "Check densenet value:",
        check_output(mge_model, model, input_shape=(1, 3, 224, 224), rtol=1e-2)
    )
