"""Independent small regression checks of generalized ring indexing."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ac_model import ACModel
from restoration import Environment, active_edges, choose_search, root_edge
from contingent_planner import ContingentPlanner, run_contingent
from run_stage1 import crosscheck

NETWORKS=json.loads((ROOT/'data/ee/networks.json').read_text())
CASES=json.loads((ROOT/'data/ee/core_cases.json').read_text())
SELECTED=[s for s in CASES if s['n_stations'] in (8,10) and '_R0_' in s['id'] and s['source_cap_mw']==75]


@pytest.mark.parametrize('scenario',SELECTED,ids=lambda s:s['id'])
def test_generalized_initial_partition_and_all_nominal_steps(scenario):
    n=scenario['n_stations'];case=NETWORKS[str(n)]
    f=int(scenario['fault_edge'][1:]);mid=n//2
    env=Environment(ACModel(case,scenario),scenario)
    expected={f'd{i}' for i in (range(mid,n) if f<mid else range(mid))}
    assert set(env.state.enabled)==expected
    expected_root='l0' if f<mid else f'l{n}'
    assert root_edge(scenario['fault_edge'],case)==expected_root
    edges=active_edges(case,env.state,scenario['fault_edge'])
    assert f'l{mid}' not in edges
    assert expected_root not in edges
    assert scenario['fault_edge'] not in edges
    planner=ContingentPlanner(env.model,0.)
    for _ in range(100):
        if env.t>=scenario['horizon_s']:
            break
        obs=env.observe()
        assert planner(obs)==choose_search(env.model,obs)
        env.step(planner.last_plan.action)
    assert env.unsafe==0
    assert all(s['converged'] for _,_,s in env.snapshots)


@pytest.mark.parametrize('n',[8,10])
def test_generalized_paired_expectation_and_independent_ac(n):
    scenario=next(s for s in SELECTED if s['n_stations']==n and s['fault_edge']=='l1')
    case=NETWORKS[str(n)];p=.25;outcomes=[];prediction=None
    for refusal in (False,True):
        cfg=dict(scenario,tie_refusal_once=refusal)
        env=Environment(ACModel(case,cfg),cfg)
        env.state=replace(env.state,reported=True)
        if prediction is None:
            prediction=ContingentPlanner(env.model,p).plan(env.observe()).expected_ens_mwh
        row,_=run_contingent(env,p);outcomes.append(row['ens_mwh'])
        seen=set()
        for edges,loads,result in env.snapshots:
            key=(tuple(edges),tuple(loads))
            if key not in seen:
                crosscheck(env.ac.case,edges,loads,result);seen.add(key)
    assert (1-p)*outcomes[0]+p*outcomes[1]==pytest.approx(prediction,abs=1e-9)
