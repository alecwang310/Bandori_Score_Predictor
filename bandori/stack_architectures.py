"""Small joint stack variants; the same bounded log-gap objective for every net."""
import json
from pathlib import Path

import numpy as np

from .joint import objective


ARCHITECTURES = (
    dict(name='mlp4', hidden=(4,)), dict(name='mlp8', hidden=(8,)),
    dict(name='mlp16', hidden=(16,)), dict(name='mlp32', hidden=(32,)),
    dict(name='deep16_8', hidden=(16, 8)), dict(name='deep32_8', hidden=(32, 8)),
    dict(name='local_skip8', hidden=(8,), local_skip=True),
    dict(name='local_skip16', hidden=(16,), local_skip=True),
    dict(name='tier_shared8', hidden=(8,), shared=True),
    dict(name='tier_shared16', hidden=(16,), shared=True),
)


class StackNetwork:
    def __init__(self, hidden=(8,), seed=1729, local_skip=False, shared=False):
        self.hidden, self.seed = tuple(hidden), seed
        self.local_skip, self.shared = local_skip, shared

    def initialize(self, x, target, weights, tiers):
        self.tiers = tuple(tiers)
        w = weights.sum(axis=1) / weights.sum()
        self.mean = (x*w[:, None]).sum(axis=0)
        self.scale = np.maximum(.05, np.sqrt(((x-self.mean)**2*w[:, None]).sum(axis=0)))
        self.trained = weights.sum(axis=0) > 0
        k = len(self.tiers)
        if (self.shared or self.local_skip) and x.shape[1] != 25*k:
            raise ValueError('Expected four explicit and 21 rule/support features per tier')
        width = 51 if self.shared else x.shape[1]
        rng = np.random.default_rng(self.seed)
        self.layers, self.biases = [], []
        for hidden in self.hidden:
            self.layers.append(rng.normal(0, 1/np.sqrt(width), (width, hidden)))
            self.biases.append(np.zeros(hidden))
            width = hidden
        self.v = np.zeros((width, 1 if self.shared else k))
        self.d = np.zeros(k)
        if self.local_skip:
            self.skip = np.zeros((k, 4))
        return self

    def _input(self, z):
        if not self.shared:
            return z
        n, k = len(z), len(self.tiers)
        local = np.concatenate((z[:, :4*k].reshape(n, k, 4),
                                z[:, 4*k:].reshape(n, k, 21)), axis=2)
        context = np.broadcast_to(local.mean(axis=1)[:, None, :], local.shape)
        ranks = np.log(np.array(self.tiers, float))
        ranks = (ranks-ranks.mean())/max(ranks.std(), 1)
        rank = np.broadcast_to(ranks[None, :, None], (n, k, 1))
        return np.concatenate((local, context, rank), axis=2).reshape(n*k, 51)

    def _forward(self, z):
        activations = [self._input(z)]
        for w, b in zip(self.layers, self.biases):
            activations.append(np.tanh(activations[-1]@w+b))
        raw = activations[-1]@self.v
        if self.shared:
            raw = raw.reshape(len(z), len(self.tiers))
        raw = raw+self.d
        if self.local_skip:
            own = z[:, :4*len(self.tiers)].reshape(len(z), len(self.tiers), 4)
            raw += np.einsum('ntf,tf->nt', own, self.skip)
        return raw, activations

    def parameters(self):
        values = [p for pair in zip(self.layers, self.biases) for p in pair]+[self.v, self.d]
        return values+([self.skip] if self.local_skip else [])

    def gradients(self, z, target, weights, bins, bias_penalty, decay):
        raw, acts = self._forward(z)
        loss, grad = objective(raw, target, weights, bins, bias_penalty)
        flat_grad = grad.reshape(-1, 1) if self.shared else grad
        dv = acts[-1].T@flat_grad+decay*self.v
        dd = grad.sum(axis=0)
        dh = flat_grad@self.v.T
        grads = []
        for i in reversed(range(len(self.layers))):
            dh *= 1-acts[i+1]**2
            dw = acts[i].T@dh+decay*self.layers[i]
            db = dh.sum(axis=0)
            dh = dh@self.layers[i].T
            grads[0:0] = [dw, db]
        grads += [dv, dd]
        squared = sum(np.sum(w*w) for w in [*self.layers, self.v])
        if self.local_skip:
            own = z[:, :4*len(self.tiers)].reshape(len(z), len(self.tiers), 4)
            grads.append(np.einsum('ntf,nt->tf', own, grad)+decay*self.skip)
            squared += np.sum(self.skip*self.skip)
        return float(loss+.5*decay*squared), grads

    def fit(self, x, target, weights, bins, config):
        if not weights.sum() > 0 or not all(np.isfinite(a).all() for a in (x, target, weights)):
            raise ValueError('Invalid training arrays')
        self.initialize(x, target, weights, config.tiers)
        z = np.clip((x-self.mean)/self.scale, -6, 6)
        params = self.parameters()
        moment, variance = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        for step in range(1, config.epochs+1):
            _, gradients = self.gradients(z, target, weights, bins, config.bias_penalty, config.weight_decay)
            for i, (p, grad) in enumerate(zip(params, gradients)):
                moment[i] = .9*moment[i]+.1*grad
                variance[i] = .999*variance[i]+.001*grad*grad
                p -= config.learning_rate*(moment[i]/(1-.9**step))/(np.sqrt(variance[i]/(1-.999**step))+1e-8)
        return self

    def predict(self, x):
        raw, _ = self._forward(np.clip((x-self.mean)/self.scale, -6, 6))
        values = .5*np.tanh(raw)
        values[:, ~self.trained] = 0
        return values

    def save(self, path):
        values = dict(mean=self.mean, scale=self.scale, trained=self.trained, v=self.v, d=self.d,
            metadata=np.array(json.dumps(dict(hidden=self.hidden, seed=self.seed, shared=self.shared,
                                             local_skip=self.local_skip, tiers=self.tiers))))
        for i, (w, b) in enumerate(zip(self.layers, self.biases)):
            values['w'+str(i)], values['b'+str(i)] = w, b
        if self.local_skip:
            values['skip'] = self.skip
        np.savez_compressed(Path(path), **values)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            metadata = json.loads(str(a['metadata']))
            tiers = metadata.pop('tiers')
            model = cls(**metadata)
            model.tiers = tuple(tiers)
            for name in ('mean', 'scale', 'trained', 'v', 'd'):
                setattr(model, name, a[name].copy())
            model.layers = [a['w'+str(i)].copy() for i in range(len(model.hidden))]
            model.biases = [a['b'+str(i)].copy() for i in range(len(model.hidden))]
            if model.local_skip:
                model.skip = a['skip'].copy()
        return model
