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
        torch_output = torch_module(Tensor(inputs))
        torch_output = {k: v.detach().cpu().numpy() for k, v in torch_output.items()}

        mge_output = mge_module(torch.Tensor(inputs))
        mge_output = {k: v.numpy() for k, v in mge_output.items()}

        assert len(mge_output) == len(torch_output)

        print("checking")
        for k, v in mge_output.items():
            torch_v = torch_output[k]
            if not np.allclose(torch_v, v, rtol, atol):
                import ipdb; ipdb.set_trace()
                return False
    return True


if __name__ == "__main__":
    from fcn import fcn_resnet50 as fcn
    from fcn_invis import fcn_resnet50 as mge_fcn

    model = fcn(pretrained=True)
    model.eval()
    mge_model = mge_fcn(pretrained=True)
    mge_model.eval()
    print("Check value:", check_output(mge_model, model, input_shape=(1, 3, 640, 640), rtol=1e-2, atol=1e-5, repeat=3))
