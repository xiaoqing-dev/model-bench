from modelbench.judge import combine_swap, compare_pair, judge_case
from tests.conftest import ScriptedJudge

RUBRIC = "Pick the better writing."


# --- the pure swap-combination logic (deterministic core of bias control) ---

def test_combine_decisive_a():
    # A picked in both orders: order(A,B)->"1"=A, order(B,A)->"2"=A
    assert combine_swap("1", "2") == ("A", True)


def test_combine_decisive_b():
    assert combine_swap("2", "1") == ("B", True)


def test_combine_both_tie():
    assert combine_swap("tie", "tie") == ("tie", True)


def test_combine_disagreement_is_inconsistent_tie():
    # judge picked position 1 both times: order(A,B)->A, order(B,A)->B -> conflict
    assert combine_swap("1", "1") == ("tie", False)


# --- end-to-end pairwise with scripted judges ---

async def test_identical_inputs_force_tie():
    """Same text as both sides MUST be a tie. The single most important check:
    a winner here means anonymisation/bias control is broken."""
    judge = ScriptedJudge(mode="prefer_token")
    v = await compare_pair(judge, "judge", RUBRIC, "same text", "same text")
    assert v.winner == "tie"


async def test_good_beats_bad_regardless_of_position():
    judge = ScriptedJudge(mode="prefer_token", token="GOOD")
    v = await compare_pair(judge, "judge", RUBRIC, "this is GOOD", "this is weak")
    assert v.winner == "A" and v.consistent


async def test_swap_test_neutralises_position_biased_judge():
    """A judge that always picks the first response should yield a tie after the
    swap test — proving position bias is caught, not rewarded."""
    judge = ScriptedJudge(mode="always_first")
    v = await compare_pair(judge, "judge", RUBRIC, "alpha", "beta")
    assert v.winner == "tie" and v.consistent is False


async def test_judge_case_round_robins_candidates():
    judge = ScriptedJudge(mode="prefer_token", token="GOOD")
    outputs = {"m1": "GOOD answer", "m2": "ok", "m3": "ok"}
    outcomes = await judge_case(judge, "judge", RUBRIC, "case1", outputs)
    assert len(outcomes) == 3  # C(3,2)
    # m1 should win both pairs it's in; the m2 vs m3 pair is a tie
    wins_for_m1 = [o for o in outcomes if o.winner == "m1"]
    assert len(wins_for_m1) == 2
