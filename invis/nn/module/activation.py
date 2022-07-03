#!/usr/bin/env python3

import warnings
from typing import Optional

from megengine import Parameter, Tensor

import invis as torch

from .. import functional as F  # import invis.nn.functional not work for py36
from .module import Module

# the fllowing not implemented in invis
# Threshold(Module)
# MultiheadAttention(Module)

# TODO: remove F.ensure_tensor_type


class ReLU(Module):
    r"""Applies the rectified linear unit function element-wise:
    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: ../scripts/activation_images/ReLU.png
    Examples::
        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
      An implementation of CReLU - https://arxiv.org/abs/1603.05201
        >>> m = nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input),m(-input)))
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


# Not implemented in F
class RReLU(Module):
    r"""Applies the randomized leaky rectified liner unit function, element-wise,
    as described in the paper:
    `Empirical Evaluation of Rectified Activations in Convolutional Network`_.
    The function is defined as:
    .. math::
        \text{RReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            ax & \text{ otherwise }
        \end{cases}
    where :math:`a` is randomly sampled from uniform distribution
    :math:`\mathcal{U}(\text{lower}, \text{upper})`.
     See: https://arxiv.org/pdf/1505.00853.pdf
    Args:
        lower: lower bound of the uniform distribution. Default: :math:`\frac{1}{8}`
        upper: upper bound of the uniform distribution. Default: :math:`\frac{1}{3}`
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> m = nn.RReLU(0.1, 0.3)
        >>> input = torch.randn(2)
        >>> output = m(input)
    .. _`Empirical Evaluation of Rectified Activations in Convolutional Network`:
        https://arxiv.org/abs/1505.00853
    """
    __constants__ = ['lower', 'upper', 'inplace']

    lower: float
    upper: float
    inplace: bool

    def __init__(
        self,
        lower: float = 1. / 8,
        upper: float = 1. / 3,
        inplace: bool = False
    ):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'lower={}, upper={}{}'.format(self.lower, self.upper, inplace_str)


class Hardtanh(Module):
    r"""Applies the HardTanh function element-wise
    HardTanh is defined as:
    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}
    The range of the linear region :math:`[-1, 1]` can be adjusted using
    :attr:`min_val` and :attr:`max_val`.
    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``
    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    __constants__ = ['min_val', 'max_val', 'inplace']

    min_val: float
    max_val: float
    inplace: bool

    def __init__(
        self,
        min_val: float = -1.,
        max_val: float = 1.,
        inplace: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> None:
        super(Hardtanh, self).__init__()
        if min_value is not None:
            warnings.warn("keyword argument min_value is deprecated and rename to min_val")
            min_val = min_value
        if max_value is not None:
            warnings.warn("keyword argument max_value is deprecated and rename to max_val")
            max_val = max_value

        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        assert self.max_val > self.min_val

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'min_val={}, max_val={}{}'.format(
            self.min_val, self.max_val, inplace_str
        )


class ReLU6(Hardtanh):
    r"""Applies the element-wise function:
    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: ../scripts/activation_images/ReLU6.png
    Examples::
        >>> m = nn.ReLU6()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace: bool = False):
        super(ReLU6, self).__init__(0., 6., inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class Sigmoid(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)


# Not implemented in F
class Hardsigmoid(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> m = nn.Hardsigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']

    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(Hardsigmoid, self).__init__()
        self.inplace = inplace

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.hardsigmoid(input, self.inplace)


class Tanh(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)


class SiLU(Module):
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.
    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}
    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(SiLU, self).__init__()
        self.inplace = inplace

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


# Not implemented in F
class Hardswish(Module):
    r"""Applies the hardswish function, element-wise, as described in the paper:
    `Searching for MobileNetV3`_.
    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> m = nn.Hardswish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    .. _`Searching for MobileNetV3`:
        https://arxiv.org/abs/1905.02244
    """
    __constants__ = ['inplace']

    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(Hardswish, self).__init__()
        self.inplace = inplace

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.hardswish(input, self.inplace)


# Not implemented in F
class ELU(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}
    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: ../scripts/activation_images/ELU.png
    Examples::
        >>> m = nn.ELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1., inplace: bool = False) -> None:
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.elu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)


# Not implemented in F
class CELU(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))
    More details can be found in the paper `Continuously Differentiable Exponential Linear Units`_ .
    Args:
        alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: ../scripts/activation_images/CELU.png
    Examples::
        >>> m = nn.CELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    .. _`Continuously Differentiable Exponential Linear Units`:
        https://arxiv.org/abs/1704.07483
    """
    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1., inplace: bool = False) -> None:
        super(CELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.celu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)


# Not implemented in F
class SELU(Module):
    r"""Applied element-wise, as:
    .. math::
        \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))
    with :math:`\alpha = 1.6732632423543772848170429916717` and
    :math:`\text{scale} = 1.0507009873554804934193349852946`.
    More details can be found in the paper `Self-Normalizing Neural Networks`_ .
    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(SELU, self).__init__()
        self.inplace = inplace

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.selu(input, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


# Not implemented in F
class GLU(Module):
    r"""Applies the gated linear unit function
    :math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half
    of the input matrices and :math:`b` is the second half.
    Args:
        dim (int): the dimension on which to split the input. Default: -1
    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`
    Examples::
        >>> m = nn.GLU()
        >>> input = torch.randn(4, 2)
        >>> output = m(input)
    """
    __constants__ = ['dim']
    dim: int

    def __init__(self, dim: int = -1) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.glu(input, self.dim)

    def extra_repr(self) -> str:
        return 'dim={}'.format(self.dim)


class GELU(Module):
    r"""Applies the Gaussian Error Linear Units function:
    .. math:: \text{GELU}(x) = x * \Phi(x)
    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)


# Not implemented in F
class Hardshrink(Module):
    r"""Applies the hard shrinkage function element-wise:
    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}
    Args:
        lambd: the :math:`\lambda` value for the Hardshrink formulation. Default: 0.5
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: ../scripts/activation_images/Hardshrink.png
    Examples::
        >>> m = nn.Hardshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['lambd']
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super(Hardshrink, self).__init__()
        self.lambd = lambd

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.hardshrink(input, self.lambd)

    def extra_repr(self) -> str:
        return '{}'.format(self.lambd)


class LeakyReLU(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)
    or
    .. math::
        \text{LeakyRELU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}

    Args:
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    __constants__ = ['inplace', 'negative_slope']
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace  # inplace not implemented

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)


class LogSigmoid(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.logsigmoid(input)


class Softplus(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))
    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.
    For numerical stability the implementation reverts to the linear function
    when :math:`input \times \beta > threshold`.
    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(Softplus, self).__init__()
        # TODO: check beta and threshold in F
        self.beta = beta
        self.threshold = threshold

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


# Not implemented in F
class Softshrink(Module):
    r"""Applies the soft shrinkage function elementwise:
    .. math::
        \text{SoftShrinkage}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` (must be no less than zero) value
            for the Softshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: ../scripts/activation_images/Softshrink.png

    Examples::
        >>> m = nn.Softshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['lambd']
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super(Softshrink, self).__init__()
        self.lambd = lambd

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.softshrink(input, self.lambd)

    def extra_repr(self) -> str:
        return str(self.lambd)


class PReLU(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)
    or
    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        ax, & \text{ otherwise }
        \end{cases}
    Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()`
    uses a single parameter :math:`a` across all input channels.
    If called with `nn.PReLU(nChannels)`, a separate :math:`a` is used for each input channel.
    .. note::
        weight decay should not be used when learning :math:`a` for good performance.
    .. note::
        Channel dim is the 2nd dim of input. When input has dims < 2, then there is
        no channel dim and the number of channels = 1.

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are legitimate:
            1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        self.num_parameters = num_parameters
        super(PReLU, self).__init__()
        # TODO: check later
        self.weight = Parameter(torch.empty(num_parameters).fill_(init))

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input, self.weight)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


# Not implemented in F
class Softsign(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: ../scripts/activation_images/Softsign.png
    Examples::
        >>> m = nn.Softsign()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.softsign(input)


# Not implemented in F
class Tanhshrink(Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Tanhshrink}(x) = x - \tanh(x)
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: ../scripts/activation_images/Tanhshrink.png
    Examples::
        >>> m = nn.Tanhshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.tanhshrink(input)


# Not implemented in F
class Softmin(Module):
    r"""Applies the Softmin function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range `[0, 1]` and sum to 1.
    Softmin is defined as:
    .. math::
        \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}
    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input
    Args:
        dim (int): A dimension along which Softmin will be computed (so every slice
            along dim will sum to 1).
    Returns:
        a Tensor of the same dimension and shape as the input, with
        values in the range [0, 1]
    Examples::
        >>> m = nn.Softmin()
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(Softmin, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.softmin(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)


class Softmax(Module):
    r"""Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.
    Softmax is defined as:
    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    When the input Tensor is a sparse tensor then the unspecifed
    values are treated as ``-inf``.
    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input
    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]
    Args:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).
    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use `LogSoftmax` instead (it's faster and has better numerical properties).
    """
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(Softmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, self.dim)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)


# TODO: check this later
class Softmax2d(Module):
    r"""Applies SoftMax over features to each spatial location.
    When given an image of ``Channels x Height x Width``, it will
    apply `Softmax` to each location :math:`(Channels, h_i, w_j)`
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]
    """

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        assert input.dim() == 4, 'Softmax2d requires a 4D tensor as input'
        return F.softmax(input, 1)


class LogSoftmax(Module):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
    input Tensor. The LogSoftmax formulation can be simplified as:
    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)
    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input
    Args:
        dim (int): A dimension along which LogSoftmax will be computed.
    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)
    """
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    @F.ensure_tensor_type
    def forward(self, input: Tensor) -> Tensor:
        return F.logsoftmax(input, self.dim)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)
