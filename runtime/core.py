"""Forward passes and loading of frozen linear and neural weights."""
from pathlib import Path
import json
import numpy as np

class JointRidge:

    def predict(self, x):
        z = np.column_stack((np.clip((x - self.mean) / self.scale, -6, 6), np.ones(len(x))))
        result = np.clip(z @ self.coefficients, -0.5, 1.0)
        result[:, ~self.trained] = 0
        return result

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as arrays:
            for k in arrays.files:
                setattr(model, k, arrays[k].copy())
        return model

class StackNetwork:

    def __init__(self, hidden=(8,), seed=1729, local_skip=False, shared=False):
        self.hidden, self.seed = (tuple(hidden), seed)
        self.local_skip, self.shared = (local_skip, shared)

    def _input(self, z):
        if not self.shared:
            return z
        n, k = (len(z), len(self.tiers))
        local = np.concatenate((z[:, :4 * k].reshape(n, k, 4), z[:, 4 * k:].reshape(n, k, 21)), axis=2)
        context = np.broadcast_to(local.mean(axis=1)[:, None, :], local.shape)
        ranks = np.log(np.array(self.tiers, float))
        ranks = (ranks - ranks.mean()) / max(ranks.std(), 1)
        rank = np.broadcast_to(ranks[None, :, None], (n, k, 1))
        return np.concatenate((local, context, rank), axis=2).reshape(n * k, 51)

    def _forward(self, z):
        activations = [self._input(z)]
        for w, b in zip(self.layers, self.biases):
            activations.append(np.tanh(activations[-1] @ w + b))
        raw = activations[-1] @ self.v
        if self.shared:
            raw = raw.reshape(len(z), len(self.tiers))
        raw = raw + self.d
        if self.local_skip:
            own = z[:, :4 * len(self.tiers)].reshape(len(z), len(self.tiers), 4)
            raw += np.einsum('ntf,tf->nt', own, self.skip)
        return (raw, activations)

    def predict(self, x):
        raw, _ = self._forward(np.clip((x - self.mean) / self.scale, -6, 6))
        values = 0.5 * np.tanh(raw)
        values[:, ~self.trained] = 0
        return values

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            metadata = json.loads(str(a['metadata']))
            tiers = metadata.pop('tiers')
            model = cls(**metadata)
            model.tiers = tuple(tiers)
            for name in ('mean', 'scale', 'trained', 'v', 'd'):
                setattr(model, name, a[name].copy())
            model.layers = [a['w' + str(i)].copy() for i in range(len(model.hidden))]
            model.biases = [a['b' + str(i)].copy() for i in range(len(model.hidden))]
            if model.local_skip:
                model.skip = a['skip'].copy()
        return model

class MediumStack(StackNetwork):

    def _dense_input(self, z):
        if not self.shared:
            return z
        n, k = (len(z), len(self.tiers))
        q = (z.shape[1] - 25 * k) // k
        if q < 0 or z.shape[1] != k * (25 + q):
            raise ValueError('Invalid tier-panel width')
        local = np.concatenate((z[:, :4 * k].reshape(n, k, 4), z[:, 4 * k:25 * k].reshape(n, k, 21), z[:, 25 * k:].reshape(n, k, q)), axis=2)
        context = np.broadcast_to(local.mean(axis=1)[:, None, :], local.shape)
        rank = np.log(np.asarray(self.tiers, float))
        rank = (rank - rank.mean()) / max(rank.std(), 1)
        return np.concatenate((local, context, np.broadcast_to(rank[None, :, None], (n, k, 1))), axis=2).reshape(n * k, -1)

    def clear_cache(self):
        self._cached_z = None
        self._cached_input = None

    def _input(self, z):
        if getattr(self, '_cached_z', None) is not z:
            self._cached_z = z
            self._cached_input = self._dense_input(z)[:, self.active]
        return self._cached_input

    @classmethod
    def load(cls, path):
        model = super().load(path)
        with np.load(path, allow_pickle=False) as a:
            model.active = a['active'].copy()
            model.dense_parameters = int(a['dense_parameters'])
        model.clear_cache()
        return model
