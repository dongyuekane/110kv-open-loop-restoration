"""Causal one-refusal contingent restoration planning on the unchanged engine.

The Bernoulli prior is a declared synthetic sensitivity parameter, not an
estimated equipment failure rate. All actions use DecisionModel.actions().
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import math
import time

from restoration import nominal_transition, physical_duration


@dataclass(frozen=True)
class Plan:
    action: str
    expected_ens_mwh: float
    expected_commands: float
    cached_states: int
    first_attempt_pending: bool


class ContingentPlanner:
    """An episode-local policy with one bit of publicly observed history.

    An initial tie-close succeeds with probability 1-p, or refuses with p.
    The refusal produces the existing tie_locked state. Rearm and subsequent
    close are deterministic, matching the benchmark's one-refusal semantics.
    Successful closure or observed refusal consumes the initial opportunity.
    No scenario, hidden refusal flag, or future acknowledgement is accepted.
    """

    def __init__(self, model, refusal_probability):
        p = float(refusal_probability)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError('refusal_probability must be finite and in [0,1]')
        self.model = model
        self.p = p
        self.first_attempt_pending = True
        self.last_plan = None

    def plan(self, obs):
        # Both facts arrive through the observation. This memory survives rearm.
        if obs.state.tie_locked or obs.state.tie_closed:
            self.first_attempt_pending = False
        if not obs.state.reported:
            result = Plan('wait', float('nan'), 0.0, 0, self.first_attempt_pending)
            self.last_plan = result
            return result
        remaining = round(obs.horizon_s-obs.elapsed_s, 6)
        if remaining < 0:
            raise ValueError('observation is beyond horizon')
        fault = obs.fault_edge

        @lru_cache(None)
        def solve(state, time_left, risk_pending):
            unserved = self.model.total_mw-self.model.electrical(state, fault)['served_mw']
            best = (unserved*time_left/3600., 0.0, 'stop')
            view = replace(obs, state=state, elapsed_s=obs.horizon_s-time_left)
            for action in self.model.actions(view):
                duration = physical_duration(action)
                if duration > time_left:
                    continue
                ack = min(obs.nominal_ack_s, time_left-duration)
                dt = duration+ack
                success = nominal_transition(state, action)
                pending_after = risk_pending and action != 'close_tie'
                if action == 'close_tie' and risk_pending and self.p > 0:
                    refused = replace(state, tie_locked=True)
                    outcomes = [(1-self.p, success, False), (self.p, refused, False)]
                else:
                    outcomes = [(1.0, success, pending_after)]
                future_ens = 0.0
                future_commands = 0.0
                for probability, nxt, next_pending in outcomes:
                    if probability == 0:
                        continue
                    after = self.model.total_mw-self.model.electrical(nxt, fault)['served_mw']
                    if dt >= time_left:
                        tail, number = 0.0, 0.0
                    else:
                        tail, number, _ = solve(nxt, round(time_left-dt, 6), next_pending)
                    # Keep arithmetic order identical to the nominal engine at p=0.
                    future_ens += probability*(unserved*duration/3600.+after*ack/3600.+tail)
                    future_commands += probability*number
                value = (future_ens, 1.0+future_commands, action)
                if value < best:
                    best = value
            return best

        expected, number, action = solve(obs.state, remaining, self.first_attempt_pending)
        result = Plan(action, expected, number, solve.cache_info().currsize,
                      self.first_attempt_pending)
        self.last_plan = result
        return result

    def __call__(self, obs):
        return self.plan(obs).action


def run_contingent(env, refusal_probability):
    """Roll out solely through the existing observe()/step() engine interface.

    Environment.run is retained unchanged for every original comparator.
    Timing includes primary AC checks and dynamic programming, with a fresh
    AC model/cache supplied by the runner for each policy and realization.
    """
    planner = ContingentPlanner(env.model, refusal_probability)
    start = time.perf_counter()
    decision_time = 0.0
    plans = []
    for _ in range(200):
        if env.t >= env.scenario['horizon_s']:
            break
        tick = time.perf_counter()
        action = planner(env.observe())
        decision_time += time.perf_counter()-tick
        if env.state.reported:
            plans.append(dict(time_s=env.t, action=action,
                expected_ens_mwh=planner.last_plan.expected_ens_mwh,
                expected_commands=planner.last_plan.expected_commands,
                cached_states=planner.last_plan.cached_states,
                first_attempt_pending=planner.last_plan.first_attempt_pending))
        env.step(action)
    else:
        raise RuntimeError('step limit')
    result = env.result()
    row = dict(scenario=env.scenario['id'], event_class=env.scenario['event_class'],
        seed=env.scenario['seed'], policy=f'contingent_p{refusal_probability:g}',
        assumed_refusal_probability=float(refusal_probability), ens_mwh=env.ens,
        served_fraction=result['served_mw']/env.model.total_mw,
        recovered_interrupted_fraction=1-(env.model.total_mw-result['served_mw'])/env.initial_unserved_mw,
        full_restoration_time_s=env.last_pickup if abs(env.model.total_mw-result['served_mw'])<1e-8 else None,
        final_unserved_mw=env.model.total_mw-result['served_mw'],
        time_to_last_pickup_s=env.last_pickup, commands=env.commands,
        failed_commands=env.failed, unsafe_executed_states=env.unsafe,
        decision_wall_s=decision_time, run_wall_s=time.perf_counter()-start,
        decision_calls=len(plans), max_planning_states=max((x['cached_states'] for x in plans), default=0))
    return row, plans
