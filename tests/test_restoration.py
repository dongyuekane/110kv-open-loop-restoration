"""Independent state/history, timing and electrical-semantic regressions.

Run in the project, or set POWERAI_PROJECT when running this reviewed copy.
These tests use synthetic data only and do not claim relay/EMT validation.
"""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(os.environ.get('POWERAI_PROJECT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT/'src'))
from ac_model import ACModel, components
from restoration import (Environment, Observation, State, DecisionModel,
    active_edges, root_edge, choose_rule, choose_search, physical_violations)


@pytest.fixture(scope='module')
def case():
    return json.loads((ROOT/'data/network.json').read_text())


def scenario(**updates):
    value=dict(id='independent_test', seed=42, fault_edge='l0', event_class='test',
        load_scale=[1.]*6, source_cap_mw=100., fault_report_s=5.,
        acknowledgement_s=5., nominal_ack_s=5., tie_refusal_once=False,
        missing_isolation_ack=False, horizon_s=1800.)
    value.update(updates)
    return value


def ready(env, *, tie=False):
    env.state=replace(env.state, reported=True, isolation=(1,1),
                      protection_reset=True, tie_closed=tie)
    env.record('test_ready')
    return env


def test_same_topology_different_protection_history(case):
    sc=scenario(fault_edge='l1');env=Environment(ACModel(case,sc),sc)
    before=replace(env.state,reported=True,isolation=(1,1))
    after=replace(before,protection_reset=True)
    ob=Observation(before,100.,1800.,'l1','l0',5.)
    oa=replace(ob,state=after)
    assert active_edges(case,before,'l1')==active_edges(case,after,'l1')
    assert env.model.electrical(before,'l1')['served_mw']==env.model.electrical(after,'l1')['served_mw']
    assert not env.model.admissibility(ob,'close_tie')[0]
    assert env.model.admissibility(oa,'close_tie')[0]


@pytest.mark.parametrize('isolation',[(0,0),(1,0),(0,1),(2,1),(1,2)])
def test_incomplete_isolation_blocks_energization(case,isolation):
    sc=scenario(fault_edge='l1');env=Environment(ACModel(case,sc),sc)
    state=replace(env.state,reported=True,isolation=isolation,protection_reset=True)
    obs=replace(env.observe(),state=state,fault_edge='l1')
    assert not env.model.admissibility(obs,'close_tie')[0]
    assert not env.model.admissibility(obs,'close_root')[0]


def test_removing_fault_from_ac_is_not_confirmed_isolation(case):
    sc=scenario(fault_edge='l1');env=Environment(ACModel(case,sc),sc)
    raw=replace(env.state,reported=True,tie_closed=True)
    electrical=env.model.electrical(raw,'l1')
    assert electrical['feasible']
    assert any(x.startswith('energized_fault_contact_') for x in physical_violations(case,raw,'l1',electrical))
    assert not env.model.admissibility(replace(env.observe(),state=replace(raw,tie_closed=False),fault_edge='l1'),'close_tie')[0]


def test_missing_ack_requires_confirmation_and_reset(case):
    sc=scenario(missing_isolation_ack=True);env=Environment(ACModel(case,sc),sc)
    env.state=replace(env.state,reported=True)
    env.step('isolate_0')
    assert env.state.isolation==(2,0)
    assert env.t==50.
    assert 'confirm_0' in env.model.actions(env.observe())
    assert not env.model.admissibility(env.observe(),'reset_protection')[0]
    env.step('isolate_1')
    assert env.state.isolation==(2,1)
    assert not env.model.admissibility(env.observe(),'reset_protection')[0]
    env.step('confirm_0')
    assert env.state.isolation==(1,1)
    env.step('reset_protection')
    assert env.state.protection_reset


def test_refusal_latches_without_energizing_then_rearms(case):
    sc=scenario(tie_refusal_once=True);env=ready(Environment(ACModel(case,sc),sc))
    old=env.result()['served_mw']
    env.step('close_tie')
    assert env.state.tie_locked and not env.state.tie_closed
    assert env.result()['served_mw']==old and env.failed==1
    assert not env.model.admissibility(env.observe(),'close_tie')[0]
    env.step('rearm_tie')
    assert not env.state.tie_locked
    env.step('close_tie')
    assert env.state.tie_closed and env.failed==1 and env.commands==3
    assert env.t==115.


def test_no_future_event_fields_reachable_from_policy_model(case):
    sc=scenario(tie_refusal_once=True,acknowledgement_s=40.,fault_report_s=300.)
    ac=ACModel(case,sc);env=Environment(ac,sc)
    hidden={'scenario','fault_report_s','acknowledgement_s','tie_refusal_once','missing_isolation_ack'}
    assert hidden.isdisjoint(vars(env.model))
    assert hidden.isdisjoint(vars(env.model.ac))
    assert hidden.isdisjoint(vars(env.observe()))
    assert env.observe().fault_edge is None


def test_equal_observations_choose_equal_actions_despite_hidden_futures(case):
    a=scenario();b=scenario(acknowledgement_s=40.,tie_refusal_once=True,missing_isolation_ack=True)
    ea=ready(Environment(ACModel(case,a),a));eb=ready(Environment(ACModel(case,b),b))
    assert ea.observe()==eb.observe()
    assert choose_rule(ea.model,ea.observe())==choose_rule(eb.model,eb.observe())
    assert choose_search(ea.model,ea.observe())==choose_search(eb.model,eb.observe())


def test_report_is_not_available_before_arrival(case):
    sc=scenario(fault_report_s=10.)
    env=Environment(ACModel(case,sc),sc)
    assert env.observe().fault_edge is None and env.model.actions(env.observe())==['wait']
    env.step('wait')
    assert env.observe().fault_edge is None
    env.step('wait')
    assert env.observe().fault_edge=='l0'


@pytest.mark.parametrize('fault',['l0','l1','l2','l4','l5','l6'])
def test_no_restoration_exact_ens_and_fault_side(case,fault):
    sc=scenario(fault_edge=fault,horizon_s=120.)
    env=Environment(ACModel(case,sc),sc)
    unserved=env.model.total_mw-env.result()['served_mw']
    expected=(27. if int(fault[1:])<3 else 30.)*120/3600
    assert unserved==(27. if int(fault[1:])<3 else 30.)
    r=env.run('no_restoration')
    assert r['ens_mwh']==pytest.approx(expected,abs=1e-10)
    assert r['commands']==0 and r['unsafe_executed_states']==0


def test_ens_uses_new_supply_during_ack_wait(case):
    sc=scenario(acknowledgement_s=40.)
    env=ready(Environment(ACModel(case,sc),sc),tie=True)
    before=env.model.total_mw-env.result()['served_mw']
    env.step('pickup_d0')
    after=env.model.total_mw-env.result()['served_mw']
    assert env.t==50.
    assert after==before-7.
    assert env.ens==pytest.approx((before*10+after*40)/3600,abs=1e-10)
    assert env.trace[-2]['time_s']==10.


@pytest.mark.parametrize('ack',[1.,40.])
def test_actual_ack_cannot_cancel_physical_completion_before_horizon(case,ack):
    sc=scenario(acknowledgement_s=ack,horizon_s=12.)
    env=ready(Environment(ACModel(case,sc),sc),tie=True)
    before=env.model.total_mw-env.result()['served_mw']
    env.step('pickup_d0')
    assert 'd0' in env.state.enabled and env.commands==1
    assert env.trace[-2]['time_s']==10.
    env.advance(sc['horizon_s']-env.t)
    assert env.ens==pytest.approx((before*10+(before-7)*2)/3600,abs=1e-10)


def test_search_counts_pickup_benefit_before_late_ack_at_horizon(case):
    sc=scenario(horizon_s=12.,nominal_ack_s=5.)
    env=ready(Environment(ACModel(case,sc),sc),tie=True)
    action=choose_search(env.model,env.observe())
    assert action.startswith('pickup_'), 'Physical completion at 10 s improves ENS before the 12 s horizon even though nominal acknowledgement arrives at 15 s.'


def test_unfinished_physical_command_does_not_energize(case):
    sc=scenario(horizon_s=8.)
    env=ready(Environment(ACModel(case,sc),sc),tie=True)
    before=env.model.total_mw-env.result()['served_mw']
    env.step('pickup_d0')
    assert 'd0' not in env.state.enabled and env.commands==1
    assert env.ens==pytest.approx(before*8/3600,abs=1e-10)
    assert 'pending_physical_at_horizon' in env.trace[-1]['event']


def test_trace_reintegrates_ens_for_actual_policy(case):
    sc=scenario(tie_refusal_once=True,missing_isolation_ack=True,acknowledgement_s=40.)
    env=Environment(ACModel(case,sc),sc);r=env.run('fixed_rule')
    integrated=sum((env.model.total_mw-a['served_mw'])*(b['time_s']-a['time_s'])/3600
        for a,b in zip(env.trace[:-1],env.trace[1:]))
    assert r['ens_mwh']==pytest.approx(integrated,abs=1e-10)
    assert all(a['time_s']<=b['time_s'] for a,b in zip(env.trace[:-1],env.trace[1:]))


def test_two_live_sources_may_not_be_paralleled(case):
    edges=[e['id'] for e in case['edges']]
    with pytest.raises(ValueError,match='unintended_source_parallel'):
        components(case,edges,('SA','SB'))
    r=ACModel(case,scenario()).solve(edges,[x['id'] for x in case['loads']])
    assert not r['feasible'] and 'unintended_source_parallel' in r['violations']


def test_energized_mesh_is_rejected(case):
    c=deepcopy(case);c['edges'].append(dict(id='mesh_edge',kind='line',**{'from':'H0','to':'H2'}))
    edges=[e['id'] for e in c['edges'] if e['id']!='l3']
    with pytest.raises(ValueError,match='energized_mesh'):
        components(c,edges,('SA',))


def test_source_budget_blocks_pickup(case):
    sc=scenario(source_cap_mw=35.)
    env=ready(Environment(ACModel(case,sc),sc),tie=True)
    good,reason=env.model.admissibility(env.observe(),'pickup_d0')
    assert not good and 'source_budget:SB' in reason


def test_transformer_rating_is_checked(case):
    c=deepcopy(case);next(e for e in c['edges'] if e['id']=='t0')['sn_mva']=5.
    ac=ACModel(c,scenario())
    r=ac.solve([e['id'] for e in c['edges'] if e['id']!='l3'],[l['id'] for l in c['loads']])
    assert not r['feasible'] and 'loading:t0' in r['violations']


def test_line_ampacity_is_checked(case):
    c=deepcopy(case);next(e for e in c['edges'] if e['id']=='l0')['max_i_ka']=.05
    ac=ACModel(c,scenario())
    r=ac.solve([e['id'] for e in c['edges'] if e['id']!='l3'],[l['id'] for l in c['loads']])
    assert r['converged'] and not r['feasible'] and 'loading:l0' in r['violations']


def test_disconnected_enabled_load_is_not_counted_as_served(case):
    sc=scenario();ac=ACModel(case,sc)
    initial=Environment(ac,sc).state
    all_enabled=replace(initial,enabled=tuple(l['id'] for l in case['loads']))
    r=ac.solve(active_edges(case,all_enabled,'l0'),all_enabled.enabled)
    assert r['converged'] and r['served_mw']==pytest.approx(30.)
    assert all('L'+str(i) not in r['live_buses'] for i in range(3))


def test_voltage_limit_is_checked(case):
    c=deepcopy(case);c['voltage_limits_pu']=[1.019,1.05]
    ac=ACModel(c,scenario())
    r=ac.solve([e['id'] for e in c['edges'] if e['id']!='l3'],[l['id'] for l in c['loads']])
    assert not r['feasible'] and any(x.startswith('voltage:') for x in r['violations'])
