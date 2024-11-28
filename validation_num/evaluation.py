# Identifiability Guarantees For Time Series Representation via Contrastive Sparsity-inducing
# Copyright 2024, ICLR 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:09:35 2023

@author: dxu3
"""
from itertools import permutations
import numpy as np
from scipy.optimize import linear_sum_assignment


def MCC(z, hz, k):

    z = z.detach().cpu().numpy()
    hz = hz.detach().cpu().numpy()
    cor_abs = np.abs(np.corrcoef(z.T, hz.T))[0:k, k:]
    # print(cor_abs)

    assignments = linear_sum_assignment(-1 * cor_abs)

    maxcor = cor_abs[assignments].sum()
    return maxcor, cor_abs


def reorder(A, d):

    B = A * 1.0

    mind = linear_sum_assignment(-1 * A)[1]

    B = np.delete(B, mind, 1)

    Ao = []
    Ao = np.expand_dims(Ao, 0)
    Ao = np.repeat(Ao, d, axis=0)

    # Order the latent variables such that latent variables displaying the highest correlation with the same source feature are together
    for i in range(d):

        Ai = np.array(A[:, mind[i]], ndmin=2).T
        Ao = np.concatenate((Ao, Ai), axis=1)

    Ao = np.concatenate((Ao, B), axis=1)

    return Ao
