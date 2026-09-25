"""Prespecified temporal splits. Confirmation scores the WHOLE historical ensemble."""
from dataclasses import dataclass, asdict
import numpy as np

TEMPLATE = [(37, 40, 44), (47, 50, 54), (52, 55, 59), (57, 60, 64), (62, 65, 70)]


@dataclass(frozen=True)
class Job:
    arm: str
    seed: int
    fold: int
    origin: int
    train_end: int
    es_start: int
    es_end: int
    score_start: int | None = None
    score_end: int | None = None

    @property
    def key(self):
        return f'{self.arm}_o{self.origin}_f{self.fold}_s{self.seed}'

    def masks(self, months):
        m = np.asarray(months)
        train = (m >= 0) & (m <= self.train_end)
        es = (m >= self.es_start) & (m <= self.es_end)
        score = ((m >= self.score_start) & (m <= self.score_end)
                 if self.score_start is not None else es.copy())
        if not train.any() or not es.any() or not score.any():
            raise ValueError(f'Empty split: {self.key}')
        if self.train_end >= self.es_start - 1 or self.es_end > self.origin:
            raise ValueError('Broken train/ES embargo or future ES')
        if np.any(train & es) or (self.score_start is not None and
                                 (self.score_start <= self.origin or np.any((train | es) & score))):
            raise ValueError('Temporal leakage')
        return train, es, score


def jobs(config):
    if config['mode'] not in ('screen', 'confirm', 'final'):
        raise ValueError('mode must be screen, confirm or final')
    seeds, arms = config['seeds'], config['arms']
    folds = config.get('folds', [0, 1, 2, 3, 4])
    if len(set(seeds)) != len(seeds) or len(set(arms)) != len(arms):
        raise ValueError('Duplicate seeds or arms')
    if not seeds or not arms or not folds or not set(folds) <= set(range(5)):
        raise ValueError('Invalid job selection')
    if config['mode'] in ('confirm', 'final') and set(folds) != set(range(5)):
        raise ValueError('Confirmation/final must reproduce all five temporal cutoffs')
    origins = config.get('origins', [{'origin': 59, 'score_start': 62, 'score_end': 66},
                                     {'origin': 64, 'score_start': 67, 'score_end': 70}])
    if config['mode'] != 'confirm':
        origins = [{'origin': 70}]
    intervals = [(o['score_start'], o['score_end']) for o in origins if 'score_start' in o]
    if any(a > b for a, b in intervals):
        raise ValueError('Invalid outer score window')
    if any(max(a, c) <= min(b, d) for i, (a, b) in enumerate(intervals)
           for c, d in intervals[i+1:]):
        raise ValueError('Overlapping outer score windows')
    result = []
    for outer in origins:
        shift = outer['origin'] - 70
        for seed in seeds:
            for fold in folds:
                tr, lo, hi = TEMPLATE[fold]
                # Rotate arm order deterministically; each model resets independent RNG.
                offset = (seed + fold) % len(arms)
                for arm in arms[offset:] + arms[:offset]:
                    result.append(Job(arm, seed, fold, outer['origin'], tr+shift, lo+shift,
                                      hi+shift, outer.get('score_start'), outer.get('score_end')))
    if len({j.key for j in result}) != len(result):
        raise ValueError('Duplicate job keys')
    return result
