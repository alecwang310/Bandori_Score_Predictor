"""Runtime settings for the frozen analog predictors."""
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
import json
import math

@dataclass(frozen=True)
class Config:
    tiers: tuple = ()
    half_life_days: float = 180
    characteristic_time_days: float | None = None
    post_rule_change_at: str | None = None
    post_rule_change_multiplier: float = 1.
    min_reference_events: int = 5
    max_stale_minutes: float = 60
    max_reference_gap_hours: float = 12
    final_availability_delay_hours: float = 24
    proxy_label_weight: float = .75

    def __post_init__(self):
        positive=(self.half_life_days,self.max_stale_minutes,self.max_reference_gap_hours,self.post_rule_change_multiplier)
        if any(not math.isfinite(v) or v<=0 for v in positive):raise ValueError('Invalid prediction settings')
        if self.characteristic_time_days is not None and (not math.isfinite(self.characteristic_time_days) or self.characteristic_time_days<=0):raise ValueError('Invalid age characteristic')
        if len(set(self.tiers))!=len(self.tiers) or any(t<1 for t in self.tiers):raise ValueError('Invalid tiers')
        if self.min_reference_events<2 or self.final_availability_delay_hours<0 or not 0<self.proxy_label_weight<=1:raise ValueError('Invalid reference policy')
        if self.post_rule_change_at is not None and datetime.fromisoformat(self.post_rule_change_at.replace('Z','+00:00')).tzinfo is None:raise ValueError('Rule boundary needs a timezone')

    def recency_weight(self,age_days):
        if self.characteristic_time_days is not None:return math.exp(-age_days/self.characteristic_time_days)
        return 2**(-age_days/self.half_life_days)

    def event_weight(self,age_days,event_start):
        weight=self.recency_weight(age_days)
        if self.post_rule_change_at is not None and event_start>=datetime.fromisoformat(self.post_rule_change_at.replace('Z','+00:00')).timestamp():weight*=self.post_rule_change_multiplier
        return weight

    @classmethod
    def load(cls,path):
        values=json.loads(Path(path).read_text());allowed={f.name for f in fields(cls)}
        values={k:v for k,v in values.items() if k in allowed};values['tiers']=tuple(values.get('tiers',()))
        return cls(**values)

