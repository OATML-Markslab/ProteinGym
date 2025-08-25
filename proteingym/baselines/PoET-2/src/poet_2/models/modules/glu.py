from typing import Callable

import torch
import torch.nn as nn


class GLU(nn.Module):
    """
    The code for this class is adapted from lucidrain's x-transformers and is licensed
    under the following license:

    MIT License

    Copyright (c) 2020 Phil Wang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=False)
        self.activation = activation

    def reset_parameters(self):
        self.proj.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, x = self.proj.forward(x).chunk(2, dim=-1)
        return self.activation(gate) * x


class ActivatedLinear(nn.Linear):
    def __init__(
        self, *args, activation: Callable[[torch.Tensor], torch.Tensor], **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(super().forward(x))
