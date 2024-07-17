"""Collection of mutation kernels. Global kernel is handled in gp.py"""

import torch
import torch.nn as nn
from gpytorch.kernels import Kernel

from src.data.data_utils import hellinger_distance

# Default PyTorch behaviour. Tradeoff between speed/precision.
# CDIST_COMPUTE_MODE = "use_mm_for_euclid_dist_if_necessary"
# For highest precision, use:
CDIST_COMPUTE_MODE = "donot_use_mm_for_euclid_dist"


class Kermut(Kernel):
    """Main kernel"""

    def __init__(
        self,
        conditional_probs: torch.Tensor,
        wt_sequence: torch.LongTensor,
        coords: torch.Tensor,
        h_scale: float = 1.0,
        h_lengthscale: float = 1.0,
        d_lengthscale: float = 1.0,
        p_lengthscale: float = 1.0,
        **kwargs,
    ):
        super(Kermut, self).__init__()
        # Register fixed parameters
        self.seq_len = wt_sequence.size(0) // 20
        wt_sequence = wt_sequence.view(self.seq_len, 20)
        self.register_buffer("wt_toks", torch.nonzero(wt_sequence)[:, 1])
        self.register_buffer("conditional_probs", conditional_probs.float())
        self.register_buffer(
            "hellinger", hellinger_distance(conditional_probs, conditional_probs)
        )
        self.register_buffer("coords", coords.float())

        if h_scale is not None:
            self.register_parameter(
                "_h_scale", torch.nn.Parameter(torch.tensor(h_scale))
            )
        else:
            self.register_buffer("_h_scale", None)

        self.register_parameter(
            "_h_lengthscale",
            torch.nn.Parameter(torch.tensor(h_lengthscale)),
        )
        self.register_parameter(
            "_d_lengthscale",
            torch.nn.Parameter(torch.tensor(d_lengthscale)),
        )
        self.register_parameter(
            "_p_lengthscale",
            torch.nn.Parameter(torch.tensor(p_lengthscale)),
        )
        self.transform_fn = nn.Softplus()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Reshape inputs
        x1 = x1.view(-1, self.seq_len, 20)
        x2 = x2.view(-1, self.seq_len, 20)
        # 1H to tokens
        x1_toks = torch.nonzero(x1)[:, 2].view(x1.size(0), -1)
        x2_toks = torch.nonzero(x2)[:, 2].view(x2.size(0), -1)
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position, third is AA.
        x1_idx = torch.argwhere(x1_toks != self.wt_toks)
        x2_idx = torch.argwhere(x2_toks != self.wt_toks)

        # Distance kernel
        x1_coords = self.coords[x1_idx[:, 1]]
        x2_coords = self.coords[x2_idx[:, 1]]
        distances = torch.cdist(
            x1_coords, x2_coords, p=2.0, compute_mode=CDIST_COMPUTE_MODE
        )
        k_dist = torch.exp(-self.d_lengthscale * distances)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = torch.exp(-self.h_lengthscale * hn)

        # Extract and transform probabilities
        p_x1 = self.conditional_probs[x1_idx[:, 1], x1_toks[x1_idx[:, 0], x1_idx[:, 1]]]
        p_x2 = self.conditional_probs[x2_idx[:, 1], x2_toks[x2_idx[:, 0], x2_idx[:, 1]]]
        p_x1 = torch.log(p_x1)
        p_x2 = torch.log(p_x2)
        p_diff = torch.abs(p_x1.unsqueeze(1) - p_x2.unsqueeze(0))
        k_p = torch.exp(-self.p_lengthscale * p_diff)

        # Multiply kernels
        k_mult = k_hn * k_dist * k_p

        # Sum over all mutations
        one_hot_x1 = torch.zeros(
            x1_idx[:, 0].size(0), x1_idx[:, 0].max().item() + 1
        ).to(x1.device)
        one_hot_x2 = torch.zeros(
            x2_idx[:, 0].size(0), x2_idx[:, 0].max().item() + 1
        ).to(x1.device)
        one_hot_x1.scatter_(1, x1_idx[:, 0].unsqueeze(1), 1)
        one_hot_x2.scatter_(1, x2_idx[:, 0].unsqueeze(1), 1)
        k_sum = torch.transpose(
            torch.transpose(k_mult @ one_hot_x2, 0, 1) @ one_hot_x1, 0, 1
        )

        # Scaling
        if self._h_scale is not None:
            k_sum = self.h_scale * k_sum

        return k_sum

    def get_params(self) -> dict:
        return {
            "h_scale": self.h_scale.item(),
            "h_lengthscale": self.h_lengthscale.item(),
            "d_lengthscale": self.d_lengthscale.item(),
            "p_lengthscale": self.p_lengthscale.item(),
        }

    @property
    def h_scale(self):
        return self.transform_fn(self._h_scale)

    @property
    def h_lengthscale(self):
        return self.transform_fn(self._h_lengthscale)

    @property
    def d_lengthscale(self):
        return self.transform_fn(self._d_lengthscale)

    @property
    def p_lengthscale(self):
        return self.transform_fn(self._p_lengthscale)


class Kermut_no_d(Kernel):
    """Ablation: No inter-residue distance kernel, k_d = 1"""

    def __init__(
        self,
        conditional_probs: torch.Tensor,
        wt_sequence: torch.LongTensor,
        h_scale: float = 1.0,
        h_lengthscale: float = 1.0,
        p_lengthscale: float = 1.0e-3,
        **kwargs,
    ):
        super(Kermut_no_d, self).__init__()

        # Register fixed parameters
        self.seq_len = wt_sequence.size(0) // 20
        wt_sequence = wt_sequence.view(self.seq_len, 20)
        self.register_buffer("wt_toks", torch.nonzero(wt_sequence)[:, 1])
        self.register_buffer("conditional_probs", conditional_probs)
        self.register_buffer(
            "hellinger", hellinger_distance(conditional_probs, conditional_probs)
        )

        if h_scale is not None:
            self.register_parameter(
                "_h_scale", torch.nn.Parameter(torch.tensor(h_scale))
            )
        else:
            self.register_buffer("_h_scale", None)

        self.register_parameter(
            "_h_lengthscale",
            torch.nn.Parameter(torch.tensor(h_lengthscale)),
        )
        self.register_parameter(
            "_p_lengthscale",
            torch.nn.Parameter(torch.tensor(p_lengthscale)),
        )
        self.transform_fn = nn.Softplus()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Reshape inputs
        x1 = x1.view(-1, self.seq_len, 20)
        x2 = x2.view(-1, self.seq_len, 20)
        # 1H to tokens
        x1_toks = torch.nonzero(x1)[:, 2].view(x1.size(0), -1)
        x2_toks = torch.nonzero(x2)[:, 2].view(x2.size(0), -1)
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position, third is AA.
        x1_idx = torch.argwhere(x1_toks != self.wt_toks)
        x2_idx = torch.argwhere(x2_toks != self.wt_toks)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = torch.exp(-self.h_lengthscale * hn)

        # Extract and transform probabilities
        p_x1 = self.conditional_probs[x1_idx[:, 1], x1_toks[x1_idx[:, 0], x1_idx[:, 1]]]
        p_x2 = self.conditional_probs[x2_idx[:, 1], x2_toks[x2_idx[:, 0], x2_idx[:, 1]]]
        p_x1 = torch.log(p_x1)
        p_x2 = torch.log(p_x2)
        p_diff = torch.abs(p_x1.unsqueeze(1) - p_x2.unsqueeze(0))
        k_p = torch.exp(-self.p_lengthscale * p_diff)

        # Add kernels
        k_mult = k_hn * k_p

        # Sum over all mutations
        one_hot_x1 = torch.zeros(
            x1_idx[:, 0].size(0), x1_idx[:, 0].max().item() + 1
        ).to(x1.device)
        one_hot_x2 = torch.zeros(
            x2_idx[:, 0].size(0), x2_idx[:, 0].max().item() + 1
        ).to(x1.device)
        one_hot_x1.scatter_(1, x1_idx[:, 0].unsqueeze(1), 1)
        one_hot_x2.scatter_(1, x2_idx[:, 0].unsqueeze(1), 1)
        k_sum = torch.transpose(
            torch.transpose(k_mult @ one_hot_x2, 0, 1) @ one_hot_x1, 0, 1
        )

        if self._h_scale is not None:
            k_sum = self.h_scale * k_sum

        return k_sum

    def get_params(self) -> dict:
        return {
            "h_scale": self.h_scale.item(),
            "h_lengthscale": self.h_lengthscale.item(),
            "p_lengthscale": self.p_lengthscale.item(),
        }

    @property
    def h_scale(self):
        return self.transform_fn(self._h_scale)

    @property
    def h_lengthscale(self):
        return self.transform_fn(self._h_lengthscale)

    @property
    def p_lengthscale(self):
        return self.transform_fn(self._p_lengthscale)


class Kermut_no_p(Kernel):
    """Ablation: No mutation probability kernel, k_p = 1"""

    def __init__(
        self,
        conditional_probs: torch.Tensor,
        wt_sequence: torch.LongTensor,
        coords: torch.Tensor,
        h_scale: float = 1.0,
        h_lengthscale: float = 1.0,
        d_lengthscale: float = 1.0,
        **kwargs,
    ):
        super(Kermut_no_p, self).__init__()

        # Register fixed parameters
        self.seq_len = wt_sequence.size(0) // 20
        wt_sequence = wt_sequence.view(self.seq_len, 20)
        self.register_buffer("wt_toks", torch.nonzero(wt_sequence)[:, 1])
        self.register_buffer("conditional_probs", conditional_probs.float())
        self.register_buffer(
            "hellinger", hellinger_distance(conditional_probs, conditional_probs)
        )
        self.register_buffer("coords", coords.float())

        if h_scale is not None:
            self.register_parameter(
                "_h_scale", torch.nn.Parameter(torch.tensor(h_scale))
            )
        else:
            self.register_buffer("_h_scale", None)

        self.register_parameter(
            "_h_lengthscale",
            torch.nn.Parameter(torch.tensor(h_lengthscale)),
        )
        self.register_parameter(
            "_d_lengthscale",
            torch.nn.Parameter(torch.tensor(d_lengthscale)),
        )
        self.transform_fn = nn.Softplus()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Reshape inputs
        x1 = x1.view(-1, self.seq_len, 20)
        x2 = x2.view(-1, self.seq_len, 20)
        # 1H to tokens
        x1_toks = torch.nonzero(x1)[:, 2].view(x1.size(0), -1)
        x2_toks = torch.nonzero(x2)[:, 2].view(x2.size(0), -1)
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position, third is AA.
        x1_idx = torch.argwhere(x1_toks != self.wt_toks)
        x2_idx = torch.argwhere(x2_toks != self.wt_toks)

        # Distance kernel
        x1_coords = self.coords[x1_idx[:, 1]]
        x2_coords = self.coords[x2_idx[:, 1]]
        distances = torch.cdist(
            x1_coords, x2_coords, p=2.0, compute_mode=CDIST_COMPUTE_MODE
        )
        k_dist = torch.exp(-self.d_lengthscale * distances)

        # Extract and transform Hellinger distances
        hn = self.hellinger[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
        k_hn = torch.exp(-self.h_lengthscale * hn)

        # Multiply kernels
        k_mult = k_hn * k_dist

        # Sum over all mutations
        one_hot_x1 = torch.zeros(
            x1_idx[:, 0].size(0), x1_idx[:, 0].max().item() + 1
        ).to(x1.device)
        one_hot_x2 = torch.zeros(
            x2_idx[:, 0].size(0), x2_idx[:, 0].max().item() + 1
        ).to(x1.device)
        one_hot_x1.scatter_(1, x1_idx[:, 0].unsqueeze(1), 1)
        one_hot_x2.scatter_(1, x2_idx[:, 0].unsqueeze(1), 1)
        k_sum = torch.transpose(
            torch.transpose(k_mult @ one_hot_x2, 0, 1) @ one_hot_x1, 0, 1
        )

        # Scaling
        if self._h_scale is not None:
            k_sum = self.h_scale * k_sum

        return k_sum

    def get_params(self) -> dict:
        return {
            "h_scale": self.h_scale.item(),
            "h_lengthscale": self.h_lengthscale.item(),
            "d_lengthscale": self.d_lengthscale.item(),
        }

    @property
    def h_scale(self):
        return self.transform_fn(self._h_scale)

    @property
    def h_lengthscale(self):
        return self.transform_fn(self._h_lengthscale)

    @property
    def d_lengthscale(self):
        return self.transform_fn(self._d_lengthscale)


class Kermut_no_h(Kernel):
    """Ablation: No site comparison kernel, k_H = 1"""

    def __init__(
        self,
        conditional_probs: torch.Tensor,
        wt_sequence: torch.LongTensor,
        coords: torch.Tensor,
        h_scale: float = 1.0,  # scales product kernel
        d_lengthscale: float = 1.0,
        p_lengthscale: float = 1.0,
        **kwargs,
    ):
        super(Kermut_no_h, self).__init__()

        # Register fixed parameters
        self.seq_len = wt_sequence.size(0) // 20
        wt_sequence = wt_sequence.view(self.seq_len, 20)
        self.register_buffer("wt_toks", torch.nonzero(wt_sequence)[:, 1])
        self.register_buffer("conditional_probs", conditional_probs.float())
        self.register_buffer("coords", coords.float())

        if h_scale is not None:
            self.register_parameter(
                "_h_scale", torch.nn.Parameter(torch.tensor(h_scale))
            )
        else:
            self.register_buffer("_h_scale", None)

        self.register_parameter(
            "_d_lengthscale",
            torch.nn.Parameter(torch.tensor(d_lengthscale)),
        )
        self.register_parameter(
            "_p_lengthscale",
            torch.nn.Parameter(torch.tensor(p_lengthscale)),
        )
        self.transform_fn = nn.Softplus()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Reshape inputs
        x1 = x1.view(-1, self.seq_len, 20)
        x2 = x2.view(-1, self.seq_len, 20)
        # 1H to tokens
        x1_toks = torch.nonzero(x1)[:, 2].view(x1.size(0), -1)
        x2_toks = torch.nonzero(x2)[:, 2].view(x2.size(0), -1)
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position, third is AA.
        x1_idx = torch.argwhere(x1_toks != self.wt_toks)
        x2_idx = torch.argwhere(x2_toks != self.wt_toks)

        # Distance kernel
        x1_coords = self.coords[x1_idx[:, 1]]
        x2_coords = self.coords[x2_idx[:, 1]]
        distances = torch.cdist(
            x1_coords, x2_coords, p=2.0, compute_mode=CDIST_COMPUTE_MODE
        )
        k_dist = torch.exp(-self.d_lengthscale * distances)

        # Extract and transform probabilities
        p_x1 = self.conditional_probs[x1_idx[:, 1], x1_toks[x1_idx[:, 0], x1_idx[:, 1]]]
        p_x2 = self.conditional_probs[x2_idx[:, 1], x2_toks[x2_idx[:, 0], x2_idx[:, 1]]]
        p_x1 = torch.log(p_x1)
        p_x2 = torch.log(p_x2)
        p_diff = torch.abs(p_x1.unsqueeze(1) - p_x2.unsqueeze(0))
        k_p = torch.exp(-self.p_lengthscale * p_diff)

        # Multiply kernels
        k_mult = k_dist * k_p

        # Sum over all mutations
        one_hot_x1 = torch.zeros(
            x1_idx[:, 0].size(0), x1_idx[:, 0].max().item() + 1
        ).to(x1.device)
        one_hot_x2 = torch.zeros(
            x2_idx[:, 0].size(0), x2_idx[:, 0].max().item() + 1
        ).to(x1.device)
        one_hot_x1.scatter_(1, x1_idx[:, 0].unsqueeze(1), 1)
        one_hot_x2.scatter_(1, x2_idx[:, 0].unsqueeze(1), 1)
        k_sum = torch.transpose(
            torch.transpose(k_mult @ one_hot_x2, 0, 1) @ one_hot_x1, 0, 1
        )

        # Scaling
        if self._h_scale is not None:
            k_sum = self.h_scale * k_sum

        return k_sum

    def get_params(self) -> dict:
        return {
            "h_scale": self.h_scale.item(),
            "d_lengthscale": self.d_lengthscale.item(),
            "p_lengthscale": self.p_lengthscale.item(),
        }

    @property
    def h_scale(self):
        return self.transform_fn(self._h_scale)

    @property
    def d_lengthscale(self):
        return self.transform_fn(self._d_lengthscale)

    @property
    def p_lengthscale(self):
        return self.transform_fn(self._p_lengthscale)


class Kermut_no_hp(Kernel):
    """Ablation: Neither site-comparison not mutation probability kernel, k_H = k_p = 1"""

    def __init__(
        self,
        wt_sequence: torch.LongTensor,
        coords: torch.Tensor,
        h_scale: float = 1.0,  # scales product kernel
        d_lengthscale: float = 1.0,
        **kwargs,
    ):
        super(Kermut_no_hp, self).__init__()

        # Register fixed parameters
        self.seq_len = wt_sequence.size(0) // 20
        wt_sequence = wt_sequence.view(self.seq_len, 20)
        self.register_buffer("wt_toks", torch.nonzero(wt_sequence)[:, 1])
        self.register_buffer("coords", coords.float())

        if h_scale is not None:
            self.register_parameter(
                "_h_scale", torch.nn.Parameter(torch.tensor(h_scale))
            )
        else:
            self.register_buffer("_h_scale", None)

        self.register_parameter(
            "_d_lengthscale",
            torch.nn.Parameter(torch.tensor(d_lengthscale)),
        )
        self.transform_fn = nn.Softplus()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        # Reshape inputs
        x1 = x1.view(-1, self.seq_len, 20)
        x2 = x2.view(-1, self.seq_len, 20)
        # 1H to tokens
        x1_toks = torch.nonzero(x1)[:, 2].view(x1.size(0), -1)
        x2_toks = torch.nonzero(x2)[:, 2].view(x2.size(0), -1)
        # Indices where x1 and x2 differ to the WT. First column is batch, second is position, third is AA.
        x1_idx = torch.argwhere(x1_toks != self.wt_toks)
        x2_idx = torch.argwhere(x2_toks != self.wt_toks)

        # Distance kernel
        x1_coords = self.coords[x1_idx[:, 1]]
        x2_coords = self.coords[x2_idx[:, 1]]
        distances = torch.cdist(
            x1_coords, x2_coords, p=2.0, compute_mode=CDIST_COMPUTE_MODE
        )
        k_dist = torch.exp(-self.d_lengthscale * distances)

        # Multiply kernels
        k_mult = k_dist

        # Sum over all mutations
        one_hot_x1 = torch.zeros(
            x1_idx[:, 0].size(0), x1_idx[:, 0].max().item() + 1
        ).to(x1.device)
        one_hot_x2 = torch.zeros(
            x2_idx[:, 0].size(0), x2_idx[:, 0].max().item() + 1
        ).to(x1.device)
        one_hot_x1.scatter_(1, x1_idx[:, 0].unsqueeze(1), 1)
        one_hot_x2.scatter_(1, x2_idx[:, 0].unsqueeze(1), 1)
        k_sum = torch.transpose(
            torch.transpose(k_mult @ one_hot_x2, 0, 1) @ one_hot_x1, 0, 1
        )

        # Scaling
        if self._h_scale is not None:
            k_sum = self.h_scale * k_sum

        return k_sum

    def get_params(self) -> dict:
        return {
            "h_scale": self.h_scale.item(),
            "d_lengthscale": self.d_lengthscale.item(),
        }

    @property
    def h_scale(self):
        return self.transform_fn(self._h_scale)

    @property
    def d_lengthscale(self):
        return self.transform_fn(self._d_lengthscale)

