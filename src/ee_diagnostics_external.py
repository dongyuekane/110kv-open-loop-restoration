from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter
import math

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ee/diagnostics'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
from ac_model import ACModel
from restoration import Environment, active_edges, nominal_transition, physical_violations
from independent_ac import solve_ac

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def report(env):
    while not env.state.reported:env.step('wait')

def pair(env, action, preparatory_action, label):
    a=env.state; oa=env.observe(); ra=env.result(); ta=env.t; ea=env.ens
    before=env.model.admissibility(oa,action)
    env.step(preparatory_action)
    b=env.state; rb=env.result(); after=env.model.admissibility(env.observe(),action)
    edges_a=active_edges(env.ac.case,a,env.scenario['fault_edge'])
    edges_b=active_edges(env.ac.case,b,env.scenario['fault_edge'])
    expected=(env.model.total_mw-ra['served_mw'])*(env.t-ta)/3600
    assert edges_a==edges_b and a.enabled==b.enabled
    assert ra==rb
    assert not before[0] and after[0],(env.scenario['id'],label,before,after)
    assert abs(expected-(env.ens-ea))<1e-10
    return dict(label=label,probe_action=action,preparatory_action=preparatory_action,
        before_state=asdict(a),after_state=asdict(b),before_permission=before,after_permission=after,
        same_active_edges=True,same_enabled_loads=True,same_electrical_result=True,
        preparatory_elapsed_s=env.t-ta,preparatory_ens_mwh=env.ens-ea,
        generated_by_valid_steps=True,trace=env.trace.copy())

def semantic_probes():
    nets=json.loads((ROOT/'data/ee/networks.json').read_text())
    scenarios=json.loads((ROOT/'data/ee/core_cases.json').read_text())
    out=[]
    for index,sc in enumerate(scenarios):
        case=nets[str(sc['n_stations'])];ac=ACModel(case,sc)
        env=Environment(ac,sc);report(env)
        proposed=nominal_transition(env.state,'close_tie')
        raw=env.model.electrical(proposed,sc['fault_edge'])
        hazard=dict(gate=env.model.admissibility(env.observe(),'close_tie'),
                    ac_feasible=raw['feasible'],
                    counterfactual_physical_violations=physical_violations(case,proposed,sc['fault_edge'],raw),
                    state=asdict(env.state),evaluated_not_executed=True)
        env.step('isolate_0');env.step('isolate_1')
        isolated=nominal_transition(env.state,'close_tie')
        er=env.model.electrical(isolated,sc['fault_edge'])
        reset_only=dict(gate=env.model.admissibility(env.observe(),'close_tie'),
                        ac_feasible=er['feasible'],
                        counterfactual_physical_violations=physical_violations(case,isolated,sc['fault_edge'],er),
                        state=asdict(env.state),evaluated_not_executed=True)
        p_reset=pair(env,'close_tie','reset_protection','protection_reset')
        missing=Environment(ac,dict(sc,missing_isolation_ack=True));report(missing)
        missing.step('isolate_0');missing.step('isolate_1')
        p_ack=pair(missing,'reset_protection','confirm_0','isolation_acknowledgement')
        refusal=Environment(ac,dict(sc,tie_refusal_once=True));report(refusal)
        refusal.step('isolate_0');refusal.step('isolate_1');refusal.step('reset_protection')
        refusal.step('close_tie')
        p_refusal=pair(refusal,'close_tie','rearm_tie','refusal_latch')
        assert all(x.unsafe==0 for x in [env,missing,refusal])
        assert hazard['ac_feasible'] and hazard['counterfactual_physical_violations']
        assert reset_only['ac_feasible'] and not reset_only['counterfactual_physical_violations']
        out.append(dict(scenario=sc['id'],n_stations=sc['n_stations'],fault=sc['fault_edge'],
                        pairs=[p_reset,p_ack,p_refusal],isolation_omission=hazard,reset_omission=reset_only,
                        unsafe_executed_states=env.unsafe+missing.unsafe+refusal.unsafe))
        if index%24==0:print('semantic',index+1,'/',len(scenarios),flush=True)
    dump('reachable_state_pairs.json',out)
    summary=dict(case_count=len(out),sizes=sorted({x['n_stations'] for x in out}),
        cases_by_size=dict(Counter(x['n_stations'] for x in out)),pairs=3*len(out),
        pairs_with_same_electrical_state_and_changed_permission=3*len(out),
        ac_feasible_but_fault_contact_hazard=sum(bool(x['isolation_omission']['counterfactual_physical_violations']) for x in out),
        reset_only_permission_rejections_without_model_physical_hazard=sum(not x['reset_omission']['counterfactual_physical_violations'] for x in out),
        unsafe_executed_states=sum(x['unsafe_executed_states'] for x in out),
        interpretation='Constructed reachable semantic coverage, not operating probabilities or relay-selectivity evidence')
    dump('reachable_state_pairs_summary.json',summary)
    return summary

def stage1_audit():
    case=json.loads((ROOT/'data/network.json').read_text())
    scenarios={s['id']:s for s in json.loads((ROOT/'data/scenarios.json').read_text())}
    ens_errors=[]
    for line in (ROOT/'results/stage1/traces.jsonl').read_text().splitlines():
        r=json.loads(line);sc=scenarios[r['scenario']]
        total=sum(l['p_mw']*s for l,s in zip(case['loads'],sc['load_scale']))
        tr=r['events'];expected=sum((total-a['served_mw'])*(b['time_s']-a['time_s'])/3600 for a,b in zip(tr[:-1],tr[1:]))
        assert all(b['time_s']>=a['time_s'] for a,b in zip(tr[:-1],tr[1:]))
        ens_errors.append(abs(expected-tr[-1]['ens_mwh']))
    ac_rows=[]
    for line in (ROOT/'results/stage1/electrical_states.jsonl').read_text().splitlines():
        r=json.loads(line);sc=scenarios[r['scenario']];c=deepcopy(case)
        for l,s in zip(c['loads'],sc['load_scale']):l['p_mw']*=s;l['q_mvar']*=s
        alt=solve_ac(c,r['active_edges'],r['enabled_loads'],['SA','SB'])
        assert alt['converged']
        buses={b['id']:b for b in c['buses']};edges={e['id']:e for e in c['edges']}
        vmin,vmax=c['voltage_limits_pu'];violations=[]
        for name,v in alt['bus'].items():
            if not vmin-1e-8<=v['vm_pu']<=vmax+1e-8:violations.append('voltage:'+name)
        for name,f in alt['branches'].items():
            e=edges[name]
            if e['kind']=='line':loading=max(f['i_from_ka'],f['i_to_ka'])/e['max_i_ka']*100
            else:
                i_from=e['sn_mva']/(math.sqrt(3)*buses[e['from']]['vn_kv'])
                i_to=e['sn_mva']/(math.sqrt(3)*buses[e['to']]['vn_kv'])
                loading=max(f['i_from_ka']/i_from,f['i_to_ka']/i_to)*100
            if loading>100+1e-7:violations.append('loading:'+name)
        for name,p in alt['source_p_mw'].items():
            if p>sc['source_cap_mw']+1e-7:violations.append('source_budget:'+name)
        source_error=max(abs(p-r['ac']['source_p_mw'][name]) for name,p in alt['source_p_mw'].items())
        assert sorted(violations)==sorted(r['ac']['violations'])
        ac_rows.append(dict(scenario=r['scenario'],state=r['state'],max_source_p_error_mw=source_error,
                           independent_violations=violations,matched_constraint_classification=True))
    dump('stage1_independent_constraint_checks.json',ac_rows)
    return dict(traces_reintegrated=len(ens_errors),max_ens_reintegration_error_mwh=max(ens_errors),
        electrical_states=len(ac_rows),independent_constraint_classifications_matched=len(ac_rows),
        max_source_p_error_mw=max(r['max_source_p_error_mw'] for r in ac_rows),
        limitation='Independent numerical and limit validation for the same declared balanced steady-state model; no physical calibration or relay transient validation')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--include-stage1',action='store_true');args=parser.parse_args()
    paths=[ROOT/'src/restoration.py',ROOT/'src/ac_model.py',ROOT/'src/independent_ac.py',
           ROOT/'data/ee/networks.json',ROOT/'data/ee/core_cases.json',Path(__file__)]
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    dump('diagnostic_manifest.json',dict(sha256=manifest,argv=sys.argv))
    a=semantic_probes();b=stage1_audit() if args.include_stage1 else None
    dump('diagnostic_summary.json',dict(semantic=a,stage1=b))
    print(json.dumps(dict(semantic=a,stage1=b),indent=2))
