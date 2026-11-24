"""Targeted scientific/causal tests; not paper performance evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ac_model import ACModel
from restoration import Environment, Observation, State, choose_search
from contingent_planner import ContingentPlanner, run_contingent

CASE = json.loads((ROOT/'data/network.json').read_text())
SCENARIOS = json.loads((ROOT/'data/scenarios.json').read_text())
PILOT = [SCENARIOS[i] for i in (0,12,24,36,48,60)]


class ToyModel:
    """Two electrically unconstrained blocks, for an analytical ordering test."""
    def __init__(self, root_mw=10., tie_mw=12.):
        self.weights = {'d0':root_mw, 'd1':tie_mw}
        self.total_mw = root_mw+tie_mw

    def electrical(self, state, fault):
        return {'served_mw':sum(self.weights[n] for n in state.enabled)}

    def actions(self, obs):
        s = obs.state
        actions = []
        if not s.root_closed:
            actions.append('close_root')
        elif 'd0' not in s.enabled:
            actions.append('pickup_d0')
        if s.tie_locked:
            actions.append('rearm_tie')
        elif not s.tie_closed:
            actions.append('close_tie')
        elif 'd1' not in s.enabled:
            actions.append('pickup_d1')
        return actions


def toy_obs(**kwargs):
    state = State(reported=True, isolation=(1,1), protection_reset=True)
    return Observation(state=state, elapsed_s=0, horizon_s=1800,
        fault_edge='l1', root_edge='l0', nominal_ack_s=5., **kwargs)


@pytest.mark.parametrize('bad', [-.01, 1.01, float('nan'), float('inf')])
def test_probability_validation(bad):
    with pytest.raises(ValueError):
        ContingentPlanner(ToyModel(), bad)


def test_simple_ordering_threshold_mechanism():
    # Equal normal setup = 50 s per block; refusal adds 35+45=80 s.
    # Comparing root-first vs tie-first gives p*=(12/10-1)*50/80=.125.
    # The exact DP may interleave actions, so verify actual first action too.
    assert ContingentPlanner(ToyModel(), 0.05)(toy_obs()) == 'close_tie'
    assert ContingentPlanner(ToyModel(), 0.25)(toy_obs()) == 'close_root'


def test_observed_refusal_memory_survives_rearm():
    planner = ContingentPlanner(ToyModel(), .5)
    obs = toy_obs()
    refused = replace(obs, state=replace(obs.state, tie_locked=True))
    planner(refused)
    assert not planner.first_attempt_pending
    planner(obs)  # Same electrical/protection state after a completed rearm.
    assert not planner.first_attempt_pending
    assert planner.last_plan.action == choose_search(ToyModel(), obs)


def test_equal_observation_and_history_equal_decision():
    obs = toy_obs()
    first = ContingentPlanner(ToyModel(), .25)
    second = ContingentPlanner(ToyModel(), .25)
    assert first.plan(obs) == second.plan(obs)
    refused = replace(obs, state=replace(obs.state, tie_locked=True))
    assert first.plan(refused) == second.plan(refused)
    assert first.plan(obs) == second.plan(obs)
    assert set(first.__dict__) == {'model','p','first_attempt_pending','last_plan'}


@pytest.mark.parametrize('scenario', PILOT, ids=lambda s:s['id'])
def test_zero_prior_matches_nominal_actions_and_cost(scenario):
    ac = ACModel(CASE, scenario)
    env = Environment(ac, scenario)
    planner = ContingentPlanner(env.model, 0.)
    for _ in range(100):
        if env.t >= scenario['horizon_s']:
            break
        obs = env.observe()
        assert planner(obs) == choose_search(env.model, obs)
        env.step(planner.last_plan.action)
    assert env.unsafe == 0


@pytest.mark.parametrize('scenario', PILOT, ids=lambda s:s['id'])
@pytest.mark.parametrize('p', [.05,.25,.5])
def test_exact_paired_expectation_under_matching_assumptions(scenario, p):
    # Start with the report available; no other uncertainty in this test.
    cfg = deepcopy(scenario)
    cfg.update(acknowledgement_s=cfg['nominal_ack_s'], missing_isolation_ack=False)
    outcomes = []
    predicted = None
    for refusal in (False, True):
        cfg['tie_refusal_once'] = refusal
        env = Environment(ACModel(CASE,cfg),deepcopy(cfg))
        env.state = replace(env.state,reported=True)
        if predicted is None:
            predicted = ContingentPlanner(env.model,p).plan(env.observe()).expected_ens_mwh
        row, _ = run_contingent(env,p)
        assert env.unsafe == 0
        outcomes.append(row['ens_mwh'])
    assert (1-p)*outcomes[0]+p*outcomes[1] == pytest.approx(predicted, abs=1e-9)


def test_physical_completion_before_ack_horizon_is_valued():
    model=ToyModel(root_mw=10., tie_mw=0.)
    obs=toy_obs()
    obs=replace(obs,state=replace(obs.state,root_closed=True,tie_closed=True,enabled=('d1',)),horizon_s=12.)
    plan=ContingentPlanner(model,.5).plan(obs)
    assert plan.action == 'pickup_d0'
    assert plan.expected_ens_mwh == pytest.approx(10*10/3600.)
