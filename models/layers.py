import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import copy

class Replace(Function):
    @staticmethod
    def forward(ctx, z1, z1_r, t):
        ctx.t = t
        return z1_r

    @staticmethod
    def backward(ctx, grad):
        grad_ = grad * 0
        grad_[ctx.t*20:(ctx.t+1)*20, ...] = grad[ctx.t*20:(ctx.t+1)*20, ...]  # NOTICE!: 20 denotes batch_size
        if ctx.t == -1:
            return (grad, grad, None)
        else:
            return (grad_, grad, None)


class WrapedSNNOp(nn.Module):

    def __init__(self, op, t):
        super(WrapedSNNOp, self).__init__()
        self.op = op
        self.op_ = copy.deepcopy(self.op)
        for name, parms in self.op_.named_parameters():
            parms.requires_grad_(False)
        self.t = t

    def forward(self, x):
        x_ = x.detach()
        op_parms = []
        for name, parms in self.op.named_parameters():
            op_parms.append(parms.data)
        for i, (name, parms) in enumerate(self.op_.named_parameters()):
            parms.data = op_parms[i]
        out1 = self.op(x_)
        out2 = self.op_(x)
        output = Replace.apply(out1, out2, self.t)
        return output

class SeqToANNContainer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor, tgc_t=-1):
        x_seq = x_seq.transpose(0, 1)  # B, T, * -> T, B, *
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        # if tgc_t != -1:
        #     print(tgc_t)
        return y_seq.view(y_shape).transpose(0, 1)  # T, B, * -> B, T, *

class Layer(nn.Module):
    def __init__(self,t,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            WrapedSNNOp(nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding), t),
            WrapedSNNOp(nn.BatchNorm2d(out_plane), t)
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            # spike = self.act((mem - self.thresh)*self.k)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x



# ----- For ResNet19 code -----


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel, t):
        super(tdBatchNorm, self).__init__()
        self.bn = WrapedSNNOp(nn.BatchNorm2d(out_panel), t)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y


# LIFSpike = LIF
