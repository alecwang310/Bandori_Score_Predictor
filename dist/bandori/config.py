"""Small, versioned experiment settings shared by both competitors."""
from dataclasses import asdict, dataclass
import json
import math
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Config:
    tiers: tuple[int, ...] = ()
    half_life_days: float = 180
    characteristic_time_days: float | None = None
    post_rule_change_at: str | None = None
    post_rule_change_multiplier: float = 1.0
    min_reference_events: int = 5
    max_stale_minutes: float = 60
    max_reference_gap_hours: float = 12
    final_availability_delay_hours: float = 24
    allow_terminal_labels: bool = False
    terminal_window_minutes: float = 5
    proxy_label_weight: float = .75
    min_train_events: int = 24
    tuning_events: int = 8
    calibration_events: int = 10
    test_events: int = 8
    calibration_prior_events: float = 12
    interval_alpha: float = .1
    bias_penalty: float = 3
    weight_decay: float = .003
    epochs: int = 240
    learning_rate: float = .01
    hidden_sizes: tuple[int, ...] = (4, 8)
    neural_gains: tuple[float, ...] = (.25, .5, 1)
    statistical_blends: tuple[float, ...] = (0, .25, .5, .75, 1)
    seed: int = 1729

    def __post_init__(self):
        if not math.isfinite(self.post_rule_change_multiplier) or self.post_rule_change_multiplier<=0:
            raise ValueError('Rule-change multiplier must be finite and positive')
        if self.post_rule_change_at is not None:
            boundary=datetime.fromisoformat(self.post_rule_change_at.replace('Z','+00:00'))
            if boundary.tzinfo is None:raise ValueError('Rule-change boundary needs a timezone')
        if self.characteristic_time_days is not None and (not math.isfinite(self.characteristic_time_days) or self.characteristic_time_days<=0):
            raise ValueError('Characteristic decay time must be finite and positive')
        if len(set(self.tiers))!=len(self.tiers) or any(t<1 for t in self.tiers):
            raise ValueError('Tiers must be distinct positive ranks')
        if min(self.half_life_days,self.max_stale_minutes,self.max_reference_gap_hours,self.calibration_prior_events)<=0:
            raise ValueError('Time limits and calibration prior must be positive')
        if self.min_reference_events<2 or self.min_train_events<4 or min(self.tuning_events,self.test_events,self.calibration_events)<2:
            raise ValueError('Insufficient event counts')
        if self.final_availability_delay_hours<0 or not 0<self.interval_alpha<1 or not 0<self.proxy_label_weight<=1:
            raise ValueError('Invalid availability, label weight or interval settings')
        if self.epochs<1 or self.bias_penalty<0 or self.weight_decay<0 or self.learning_rate<=0:
            raise ValueError('Invalid neural optimization settings')
        if not self.hidden_sizes or any(h<1 for h in self.hidden_sizes):
            raise ValueError('Hidden sizes must be positive')
        if not self.statistical_blends or any(not 0<=a<=1 for a in (*self.statistical_blends,*self.neural_gains)):
            raise ValueError('Mixture weights must lie in [0,1]')

    def to_dict(self):
        return asdict(self)

    def recency_weight(self,age_days):
        """Exponential age weight; characteristic time is the e-folding time."""
        if self.characteristic_time_days is not None:
            return math.exp(-age_days/self.characteristic_time_days)
        return 2**(-age_days/self.half_life_days)

    def event_weight(self,age_days,event_start):
        """Age decay times an explicit rule-era weight, keyed by actual date."""
        weight=self.recency_weight(age_days)
        if self.post_rule_change_at is not None:
            boundary=datetime.fromisoformat(self.post_rule_change_at.replace('Z','+00:00')).timestamp()
            if event_start>=boundary:weight*=self.post_rule_change_multiplier
        return weight

    @classmethod
    def load(cls,path=None):
        values=json.loads(Path(path).read_text()) if path else {}
        for field in ('tiers','hidden_sizes','neural_gains','statistical_blends'):
            if field in values:values[field]=tuple(values[field])
        return cls(**values)
