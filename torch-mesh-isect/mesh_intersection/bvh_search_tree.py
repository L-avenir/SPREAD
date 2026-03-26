from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys

import torch
import torch.nn as nn
import torch.autograd as autograd

import bvh_cuda


class BVHFunction(autograd.Function):

    max_collisions = 8

    @staticmethod
    @torch.no_grad()
    def forward(ctx, triangles):
        outputs = bvh_cuda.forward(triangles,
                                   max_collisions=BVHFunction.max_collisions)
        ctx.save_for_backward(outputs, triangles)
        return outputs

    @staticmethod
    def backward(ctx, grad_output, *args, **kwargs):
        raise NotImplementedError


class BVH(nn.Module):

    def __init__(self, max_collisions=8):
        super(BVH, self).__init__()
        self.max_collisions = max_collisions
        BVHFunction.max_collisions = self.max_collisions

    def forward(self, triangles):
        return BVHFunction.apply(triangles)
