import math
from inspect import isfunction

import torch
import torch.nn.functional as F
from torch import nn

MIN_EXPERT_CAPACITY = 4


def supported_hyperparameters():
    return {'lr', 'adam_betas_1','adam_betas_2', 'adam_eps', 'adam_weight_decay', 'eps',
            'second_threshold_train', 'second_threshold_eval', 'capacity_factor_train', 'capacity_factor_eval'}

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class Experts(nn.Module):
    def __init__(self, dim, num_experts=16, hidden_dim=None, activation=GELU):
        super().__init__()
        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)
        w1 = init_(torch.zeros(*num_experts, dim, hidden_dim))
        w2 = init_(torch.zeros(*num_experts, hidden_dim, dim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

class Top2Gating(nn.Module):
    def __init__(self, dim, num_gates, prm):
        super().__init__()
        self.eps = prm['eps']
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(dim, num_gates))
        self.second_policy_train = 'random'
        self.second_policy_eval = 'random'
        self.second_threshold_train = prm['second_threshold_train']
        self.second_threshold_eval = prm['second_threshold_eval']
        self.capacity_factor_train = 2 * prm['capacity_factor_train']
        self.capacity_factor_eval = 4 * prm['capacity_factor_eval']

    def forward(self, x, importance=None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates
        policy = self.second_policy_train if self.training else self.second_policy_eval
        threshold = self.second_threshold_train if self.training else self.second_threshold_eval
        capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating).softmax(dim=-1)
        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]

        gates_without_top_1 = raw_gates * (1. - mask_1)
        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]

        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        density_1 = mask_1.mean(dim=-2)
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)

        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        mask_1 *= (position_in_expert_1 < expert_capacity).float()
        mask_1_flat = mask_1.sum(dim=-1)
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1.sum(dim=-2, keepdim=True)
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity).float()
        mask_2_flat = mask_2.sum(dim=-1)
        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat

        combine_tensor = (
            gate_1[..., None, None] * mask_1_flat[..., None, None] * F.one_hot(index_1, num_gates)[..., None] * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            gate_2[..., None, None] * mask_2_flat[..., None, None] * F.one_hot(index_2, num_gates)[..., None] * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm):
        super().__init__()
        dim = in_shape[-1]
        num_experts = prm.get("num_experts", (4, 4))
        hidden_dim = prm.get("hidden_dim", dim * 4)
        activation = prm.get("activation", nn.ReLU)

        self.num_experts_outer, self.num_experts_inner = num_experts
        self.loss_coef = prm.get("loss_coef", 1e-2)

        # Gating networks
        self.gate_outer = Top2Gating(dim, self.num_experts_outer, prm)
        self.gate_inner = Top2Gating(dim, self.num_experts_inner, prm)

        # Experts
        self.experts = Experts(dim, num_experts=num_experts, hidden_dim=hidden_dim, activation=activation)

    def forward(self, x):
        b, n, d = x.shape
        eo, ei = self.num_experts_outer, self.num_experts_inner

        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(x)
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', x, dispatch_tensor_outer)

        importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1)
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())


        dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance)
        expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)


        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)


        expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)


        total_loss = (loss_outer + loss_inner) * self.loss_coef
        return output, total_loss

    def train_setup(self, device, prm):
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=prm.get("lr", 1e-3))
        self.criterion = nn.MSELoss()


    def learn(self, train_data):
        pass  # todo implementation
