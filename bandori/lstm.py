"""Near-bare joint raw-score LSTM (PyTorch) with causal query branches.

The main state follows an hourly grid for the entire event. Query branches
share the cell and decoder but never update the main history, so the same
computation serves any off-grid inference timestamp. One stock LSTM cell
reads the raw stream; a small shared head emits all tier corrections at
once. Training and inference run on Apple Metal (MPS) when available and
fall back to CPU. Inputs are float32.
"""
import os

# Conda ships its own libomp next to the one bundled with torch wheels; the
# two together abort at import. This override is a no-op on clean
# interpreters (e.g. the project .venv) and only matters there.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from torch import nn


def device_of(name=None):
    if name and name != 'auto':
        return torch.device(name)
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def relative_objective(raw, target, weights, bins, bias_penalty):
    """Relative error and signed-bias penalty, bounded log pace adjustment.

    raw (rows, tiers) unbounded correction; target columns are current
    score, remaining-to-close anchor and truth. Per-horizon-bin signed bias
    is penalized per tier, matching the joint-network objective.
    """
    delta = 3 * torch.tanh(raw / 3)
    future = target[..., 1] * torch.exp(delta)
    truth = torch.clamp(target[..., 2], min=1.0)
    error = (target[..., 0] + future - truth) / truth
    w = weights / torch.clamp(weights.sum(), min=1e-30)
    loss = torch.sum(w * error ** 2)
    for group in range(5):
        mask = bins == group
        if not bool(mask.any()):
            continue
        mass = w[mask].sum(dim=0)
        mean = (w[mask] * error[mask]).sum(dim=0) / torch.clamp(mass, min=1e-30)
        loss = loss + bias_penalty * torch.sum(mass * mean ** 2)
    return loss


class RawJointLSTM(nn.Module):
    def __init__(self, hidden=32, decoder=16, seed=1729, device=None):
        super().__init__()
        self.hidden, self.decoder, self.seed = hidden, decoder, seed
        self.device = device_of(device)
        self.trained = torch.ones(0, dtype=torch.bool)

    def initialize(self, input_width, side_width, tiers):
        torch.manual_seed(self.seed)
        self.cell = nn.LSTMCell(input_width, self.hidden)
        self.head = nn.Linear(self.hidden + side_width, self.decoder)
        self.out = nn.Linear(self.decoder, tiers)
        self.trained = torch.ones(tiers, dtype=torch.bool)
        return self.to(self.device)

    def _step(self, x, h, c, mask):
        nh, nc = self.cell(x, (h, c))
        mask = mask.unsqueeze(-1)
        return mask * nh + (1 - mask) * h, mask * nc + (1 - mask) * c

    def _decode(self, h, side):
        return self.out(torch.tanh(self.head(torch.cat((h, side), dim=-1))))

    def forward_tensors(self, x, side, step_mask, query_x=None, query_side=None, query_mask=None):
        """Return raw corrections (T,2,n,tiers) with queries or (T,n,tiers)."""
        length, n = x.shape[:2]
        h = torch.zeros((n, self.hidden), device=self.device)
        c = h.clone()
        main, branch = [], []
        for step in range(length):
            h, c = self._step(x[step], h, c, step_mask[step])
            main.append(self._decode(h, side[step]))
            if query_x is not None:
                qh, qc = self._step(query_x[step], h, c, query_mask[step])
                branch.append(self._decode(qh, query_side[step]))
        main = torch.stack(main)
        return torch.stack((main, torch.stack(branch)), dim=1) if query_x is not None else main

    def to_tensors(self, bundle):
        def t(key):
            return torch.as_tensor(bundle[key], dtype=torch.float32, device=self.device)
        return t('x'), t('side'), t('step_mask'), t('query_x'), t('query_side'), t('query_mask')

    def fit(self, bundle, target, weights, bins, epochs=120, learning_rate=.005, bias_penalty=3,
            decay=.001, progress=None):
        if weights.sum() <= 0 or not all(np.isfinite(a).all() for a in (bundle['x'], bundle['side'], target, weights)):
            raise ValueError('Invalid LSTM training inputs')
        self.initialize(bundle['x'].shape[-1], bundle['side'].shape[-1], target.shape[-2])
        self.trained = torch.as_tensor(weights.reshape(-1, weights.shape[-1]).sum(axis=0) > 0,
                                       dtype=torch.bool, device=self.device)
        x, side, step_mask, query_x, query_side, query_mask = self.to_tensors(bundle)
        y = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        b = torch.as_tensor(bins.reshape(-1), dtype=torch.long, device=self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=decay)
        k = y.shape[-2]
        for step in range(1, epochs + 1):
            optimizer.zero_grad()
            raw = self.forward_tensors(x, side, step_mask, query_x, query_side, query_mask)
            loss = relative_objective(raw.reshape(-1, k), y.reshape(-1, k, 3), w.reshape(-1, k), b, bias_penalty)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            if progress and (step == 1 or step % 20 == 0 or step == epochs):
                progress(step, float(loss.detach()))
        return self

    @torch.no_grad()
    def predict_corrections(self, bundle):
        x, side, step_mask, query_x, query_side, query_mask = self.to_tensors(bundle)
        raw = self.forward_tensors(x, side, step_mask, query_x, query_side, query_mask)
        result = 3 * torch.tanh(raw / 3)
        result[..., ~self.trained] = 0
        return result.cpu().numpy()

    @torch.no_grad()
    def forecast(self, main_x, main_side, query_x, query_side, slots, off_grid):
        """Bounded corrections for arbitrary-time queries on one event.

        main_x is the (T, features) hourly grid; query_x holds one row per
        query time. One causal hourly pass builds the main state; each query
        applies a single shared-cell step from its floor slot and never
        writes back. On-grid queries decode the stored grid state directly.
        """
        def t(array):
            return torch.as_tensor(array, dtype=torch.float32, device=self.device)
        x, side, qx, qs = t(main_x), t(main_side), t(query_x), t(query_side)
        length = x.shape[0]
        h = torch.zeros((1, self.hidden), device=self.device)
        c = h.clone()
        states_h, states_c = [], []
        for step in range(length):
            h, c = self._step(x[step:step + 1], h, c, torch.ones(1, device=self.device))
            states_h.append(h[0])
            states_c.append(c[0])
        slots_t = torch.as_tensor(slots, dtype=torch.long, device=self.device)
        hq = torch.stack(states_h)[slots_t]
        cq = torch.stack(states_c)[slots_t]
        ones = torch.ones(qx.shape[0], device=self.device)
        qh, _ = self._step(qx, hq, cq, ones)
        mask = t(np.asarray(off_grid, dtype=np.float32)).unsqueeze(-1)
        hq = mask * qh + (1 - mask) * hq
        raw = self._decode(hq, qs)
        delta = 3 * torch.tanh(raw / 3)
        delta[:, ~self.trained] = 0
        return delta.cpu().numpy()

    def save(self, path):
        torch.save({'hidden': self.hidden, 'decoder': self.decoder, 'seed': self.seed,
                    'input_width': self.cell.input_size if hasattr(self, 'cell') else 0,
                    'trained': self.trained.cpu(),
                    'state_dict': {k: v.cpu() for k, v in self.state_dict().items()}}, path)

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(payload['hidden'], payload['decoder'], payload['seed'], device)
        if 'state_dict' in payload and payload.get('input_width'):
            side_width = payload['state_dict']['head.weight'].shape[1] - payload['hidden']
            model.initialize(payload['input_width'], side_width, len(payload['trained']))
        model.load_state_dict(payload['state_dict'])
        model.trained = payload['trained'].to(model.device)
        return model
