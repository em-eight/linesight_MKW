""" Implements Spectral Normalization of Conv2d layers, following
    https://arxiv.org/abs/1804.04368 and the PyTorch implementation of
    torch.nn.utils.spectral_norm
"""
import torch
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)

__all__ = ["spectral_norm_conv2d", "spectral_norm"]

"""
Spectral normalization implementation from https://github.com/floringogianu/snrl
This file is a coallation of conv_spectral_norm.py and linear_spectral_norm.py from the above repository
"""
class Conv2dSpectralNorm:
    """ Approximates spectral norm of kernel weights using power itteration. """

    _version = 1

    def __init__(  # pylint: disable=bad-continuation
        self,
        name,
        n_power_iterations=1,
        eps=1e-12,
        active=True,
        leave_smaller=False,
        lipschitz_k=1,
        flow_through_norm=True,
    ):
        assert n_power_iterations >= 0, "n_power_iterations should be positive."
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self._active = bool(active)
        self._leave_smaller = bool(leave_smaller)  # do not touch the children, pervert!
        self._lipschitz_k = float(lipschitz_k)
        self._flow_through_norm = bool(flow_through_norm)

        if self._flow_through_norm:
            raise NotImplementedError("We don't have it yet.")

    def __call__(self, module, inputs):
        # eigenvetors u and v are lazy initialized.
        if getattr(module, f"{self.name}_u").ndim == 0:
            self._set_eigenvectors(module, inputs)

        # normalize weights and set them in-place.
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    def compute_weight(self, module, do_power_iteration):
        r"""Where the deed is done.
        """
        A = getattr(module, self.name + "_orig")  # this is the kernel
        u = getattr(module, self.name + "_u")  # left eigenvector
        v = getattr(module, self.name + "_v")  # right eigenvector
        sigma = getattr(module, self.name + "_sigma")  # sigma, of course
        eps = torch.tensor(self.eps, device=A.device)
        stride = module.stride
        padding = module.padding
        dilation = module.dilation

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v_ = F.conv2d(
                        u, A, stride=stride, padding=padding, dilation=dilation
                    )
                    beta = torch.max(v_.norm(), eps)
                    v = torch.div(v_, beta, out=v)

                    u_ = F.conv_transpose2d(
                        v, A, stride=stride, padding=padding, dilation=dilation
                    )

                    # this is the largest eigenvalue
                    sigma.copy_(torch.max(u_.norm(), eps))
                    # TODO: For some reason, the shape of the network is slightly mismatched. Possible rounding error when downsizing the image.
                    if u.shape != u_.shape:
                        print("Warning: During Spectral Normalization weight computation, got mismatching buffer shapes:", u_.shape, u.shape, "Now resizing..")
                        u.resize_(u_.shape).copy_(u_)
                    u = torch.div(u_, sigma, out=u)

                    # See above on why we need to clone
                    if self.n_power_iterations > 0:
                        u = u.clone(memory_format=torch.contiguous_format)
                        v = v.clone(memory_format=torch.contiguous_format)

        """if self._active:
            if self._leave_smaller:
                A = A / max(sigma.item() / self._lipschitz_k, 1)
            else:
                A = A / (sigma.item() / self._lipschitz_k)
        else:
            A = A + 0"""
        
        if self._active:
            if self._leave_smaller:
                A = A / torch.maximum(sigma / torch.tensor(self._lipschitz_k, device=sigma.device), torch.tensor(1.0, device=sigma.device))
            else:
                A = A / (sigma / torch.tensor(self._lipschitz_k, device=sigma.device))
        else:
            A = A + 0

        return A

    @torch.no_grad()
    def _set_eigenvectors(self, module, inputs):
        """ This is called once, if the `u` and `v` buffers have not been
            yet registered. We don't do this in the `apply` static method
            because we need forward time information such as input size.
        """
        w = module._parameters[f"{self.name}_orig"]
        u_shape = torch.Size((1, *inputs[0].shape[-3:]))  # eg.: (1, 1, 84, 84)
        u = F.normalize(w.new_empty(u_shape).normal_(0, 1), dim=0, eps=self.eps)

        # we do this only to get the shape.
        v_shape = F.conv2d(u, w, stride=module.stride).shape
        v = F.normalize(w.new_empty(v_shape).normal_(0, 1), dim=0, eps=self.eps)

        sigma = getattr(module, self.name + "_sigma")  # right eigenvector
        sigma.copy_(torch.max(u.norm(), torch.full_like(u.norm(), self.eps)))

        # set the buffers
        getattr(module, self.name + "_u").resize_as_(u).copy_(u)
        getattr(module, self.name + "_v").resize_as_(v).copy_(v)

    @staticmethod
    def apply(
        module,
        name,
        n_power_iterations,
        eps,
        active,
        leave_smaller=False,
        lipschitz_k=1,
        flow_through_norm=False,
    ):
        r"""Because the normalization is dependent of the input size, we
        lazy initialize some of the objects.
        """

        for _, hook in module._forward_pre_hooks.items():
            if (
                isinstance(hook, (SpectralNorm, Conv2dSpectralNorm))
                and hook.name == name
            ):
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )
        fn = Conv2dSpectralNorm(
            name,
            n_power_iterations,
            eps,
            active=active,
            leave_smaller=leave_smaller,
            lipschitz_k=lipschitz_k,
            flow_through_norm=flow_through_norm,
        )
        weight = module._parameters[name]

        # we register the Parameters to a new attribute
        delattr(module, fn.name)
        module.register_parameter(f"{fn.name}_orig", weight)
        module.register_buffer(name + "_sigma", torch.ones_like(weight.sum()))
        module.register_buffer(name + "_u", weight.new_empty(()))
        module.register_buffer(name + "_v", weight.new_empty(()))

        # we keep the old attribute around, but as a simple tensor
        # this is required because other torch stuff assumes it exists.
        setattr(module, fn.name, weight.data)
        # setattr(module, fn.name + "_sigma", fn.eps)

        # hooks
        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

    def remove(self, module):
        weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_v")
        delattr(module, self.name + "_orig")
        delattr(module, self.name + "_sigma")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))


def spectral_norm_conv2d(  # pylint: disable=bad-continuation
    module,
    name="weight",
    n_power_iterations=1,
    eps=1e-12,
    active=True,
    leave_smaller=False,
    lipschitz_k=1,
    flow_through_norm=False,
):
    r"""Applies spectral normalization to parameters of Conv2d modules.

    It is still doing SVD power itteration, but considers the special
    structure of the convolution operator. In effect, this implementation
    approximates the spectral norm of the doubly block circulant matrix.

    Example::
        >>> TODO
    """

    Conv2dSpectralNorm.apply(
        module,
        name,
        n_power_iterations,
        eps,
        active,
        leave_smaller=leave_smaller,
        lipschitz_k=lipschitz_k,
        flow_through_norm=flow_through_norm,
    )
    return module


class LinearSpectralNorm(SpectralNorm):
    def __init__(
        self,
        *args,
        active=True,
        leave_smaller=False,
        lipschitz_k=1,
        flow_through_norm=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._active = bool(active)
        self._leave_smaller = bool(leave_smaller)  # do not touch the children, pervert!
        self._lipschitz_k = float(lipschitz_k)
        self._flow_through_norm = bool(flow_through_norm)

    def compute_weight(self, module, do_power_iteration):
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        sigma = getattr(module, self.name + "_sigma")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(
                        torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
                    )
                    u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        _sigma = torch.dot(u, torch.mv(weight_mat, v))
        sigma.copy_(_sigma.data)

        if not self._flow_through_norm:
            _sigma = _sigma.detach()

        if self._active:
            if self._leave_smaller:
                weight = weight / max(_sigma / self._lipschitz_k, 1)
            else:
                weight = weight / (_sigma / self._lipschitz_k)
        else:
            weight = weight + 0

        return weight

    @staticmethod
    def apply(  # pylint: disable=bad-continuation,arguments-differ
        module,
        name,
        n_power_iterations,
        dim,
        eps,
        active,
        leave_smaller=False,
        lipschitz_k=1,
        flow_through_norm=True,
    ):
        for _k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = LinearSpectralNorm(
            name,
            n_power_iterations,
            dim,
            eps,
            active=active,
            leave_smaller=leave_smaller,
            lipschitz_k=lipschitz_k,
            flow_through_norm=flow_through_norm,
        )
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(
            fn.name + "_sigma", torch.dot(u, torch.mv(weight_mat, v).detach())
        )
        # setattr(module, fn.name + "_sigma", fn.eps)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


def spectral_norm(
    module,
    name="weight",
    n_power_iterations=1,
    eps=1e-12,
    dim=None,
    active=True,
    leave_smaller=False,
    lipschitz_k=1,
    flow_through_norm=True,
):
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    LinearSpectralNorm.apply(
        module,
        name,
        n_power_iterations,
        dim,
        eps,
        active,
        leave_smaller=leave_smaller,
        lipschitz_k=lipschitz_k,
        flow_through_norm=flow_through_norm,
    )
    return module