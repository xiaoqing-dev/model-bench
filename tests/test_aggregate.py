from modelbench.aggregate import win_rates
from modelbench.judge import PairOutcome


def test_win_rates_basic():
    outcomes = [
        PairOutcome("c1", "a", "b", winner="a", consistent=True),
        PairOutcome("c1", "a", "c", winner="a", consistent=True),
        PairOutcome("c1", "b", "c", winner="tie", consistent=True),
    ]
    standings = win_rates(outcomes)
    by_label = {s.label: s for s in standings}

    assert by_label["a"].wins == 2 and by_label["a"].losses == 0
    assert by_label["a"].win_rate == 1.0
    assert by_label["b"].wins == 0 and by_label["b"].losses == 1 and by_label["b"].ties == 1
    assert by_label["c"].losses == 1 and by_label["c"].ties == 1

    # sorted best first
    assert standings[0].label == "a"


def test_win_rate_excludes_ties():
    # a: 1 win, 1 tie -> decided=1 -> win_rate 1.0
    outcomes = [
        PairOutcome("c1", "a", "b", winner="a", consistent=True),
        PairOutcome("c2", "a", "b", winner="tie", consistent=True),
    ]
    by_label = {s.label: s for s in win_rates(outcomes)}
    assert by_label["a"].win_rate == 1.0
    assert by_label["a"].ties == 1


def test_win_rate_default_when_undecided():
    outcomes = [PairOutcome("c1", "a", "b", winner="tie", consistent=True)]
    by_label = {s.label: s for s in win_rates(outcomes)}
    assert by_label["a"].win_rate == 0.5  # nothing decided
