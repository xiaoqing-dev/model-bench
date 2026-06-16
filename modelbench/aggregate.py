"""Aggregate pairwise outcomes into a leaderboard."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Standing:
    label: str
    wins: int = 0
    losses: int = 0
    ties: int = 0

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        """Wins / decisive games. Ties excluded. 0.5 when nothing decided."""
        return self.wins / self.decided if self.decided else 0.5


def win_rates(outcomes: list) -> list:
    """Fold a list of PairOutcome into per-label standings, sorted best first."""
    table: dict = {}

    def get(label: str) -> Standing:
        if label not in table:
            table[label] = Standing(label=label)
        return table[label]

    for o in outcomes:
        a, b = get(o.a), get(o.b)
        if o.winner == o.a:
            a.wins += 1
            b.losses += 1
        elif o.winner == o.b:
            b.wins += 1
            a.losses += 1
        else:  # tie
            a.ties += 1
            b.ties += 1

    return sorted(
        table.values(),
        key=lambda s: (s.win_rate, s.wins),
        reverse=True,
    )
