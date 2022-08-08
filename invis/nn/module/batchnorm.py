#!/usr/bin/env python3

from megengine.module import BatchNorm2d as MGE_BatchNorm2d

from invis.utils.wrapper import is_torch_tensor

from .patch import patch_attribute, patch_method

__all__ = ["BatchNorm2d"]


class BatchNorm2d(MGE_BatchNorm2d):

    def __init__(self, *args, **kwargs):
        if "momentum" in kwargs:  # momentum in BatchNorm of mge is not momentum in pytorch
            kwargs["momentum"] = 1 - kwargs["momentum"]
        elif len(args) > 2:
            args = list(args)
            args[2] = 1 - args[2]
            args = tuple(args)
        super().__init__(*args, **kwargs)

        patch_attribute(self)
        if self.affine:
            self._parameters["weight"] = self.weight
            self._parameters["bias"] = self.bias
        self._buffers["running_mean"] = self.running_mean
        self._buffers["running_var"] = self.running_var

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        local_state_dict = self._state_dict(keep_var=True)
        weights_to_load = {k: state_dict[prefix + k] for _, k in local_state_dict}
        for k, v in weights_to_load.items():
            if is_torch_tensor(v):
                weights_to_load[k] = v.detach().cpu().numpy()

        for (module_type, k), var in local_state_dict.items():
            if k not in weights_to_load:
                missing_keys.append(k)
                continue

            to_be_load = weights_to_load[k]
            if var.shape != to_be_load.shape:
                if k in ("running_mean", "running_var", "bias", "weight"):
                    to_be_load = to_be_load.reshape(var.shape)
                else:
                    raise ValueError(
                        f"param `{k}` shape mismatch, should be {var.shape}, get {to_be_load.shape}"
                    )
            var._reset(
                type(var)(
                    to_be_load, dtype=to_be_load.dtype, device=var.device, no_cache=True
                )
            )


BatchNorm2d = patch_method(BatchNorm2d, patch_override=False)
