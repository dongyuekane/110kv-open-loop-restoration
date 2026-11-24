"""Run the frozen diagnostic suite; write real metrics, traces and provenance."""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

from ac_model import ACModel
from restoration import Environment, State, Observation, DecisionModel, active_edges

ROOT=Path(__file__).resolve().parents[1]


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def crosscheck(case,edges,loads,primary):
    from independent_ac import solve_ac
    alternate=solve_ac(case,edges,loads,['SA','SB'])
    if not alternate['converged']:
        raise AssertionError('Independent AC solver failed')
    v=max(abs(primary['bus'][n]['vm_pu']-alternate['bus'][n]['vm_pu']) for n in primary['bus'])
    p=max(abs(primary['branches'][n][k]-alternate['branches'][n][k])
          for n in primary['branches'] for k in ['p_from_mw','q_from_mvar','p_to_mw','q_to_mvar'])
    i=max(abs(primary['branches'][n][k]-alternate['branches'][n][k])
          for n in primary['branches'] for k in ['i_from_ka','i_to_ka'])
    if v>1e-7 or p>1e-5 or i>1e-7:
        raise AssertionError(f'Independent AC mismatch: V={v}, S={p}, I={i}')
    return dict(max_voltage_error_pu=v,max_branch_power_error_mva=p,max_current_error_ka=i,
                loss_error_mw=abs(primary['loss_mw']-alternate['loss_mw']),
                alternate_residual_pu=alternate['max_residual_pu'])


def history_probe(case,scenario):
    ac=ACModel(case,scenario); model=DecisionModel(ac)
    s=replace(Environment(ac,scenario).state,reported=True,isolation=(1,1))
    ready=replace(s,protection_reset=True)
    obs=Observation(s,100,1800,scenario['fault_edge'],'l0' if int(scenario['fault_edge'][1:])<3 else 'l6',5)
    r_obs=replace(obs,state=ready)
    same=active_edges(case,s,scenario['fault_edge'])==active_edges(case,ready,scenario['fault_edge'])
    unconfirmed=replace(ready,isolation=(2,1))
    refused=replace(ready,tie_locked=True)
    raw=replace(s,isolation=(0,0),tie_closed=True)
    return dict(scenario=scenario['id'],same_electrical_topology=same,
      before_reset=model.admissibility(obs,'close_tie'),after_reset=model.admissibility(r_obs,'close_tie'),
      missing_ack=model.admissibility(replace(r_obs,state=unconfirmed),'close_tie'),
      refusal_latch=model.admissibility(replace(r_obs,state=refused),'close_tie'),
      ac_only_fault_line_removed_would_allow_unloaded_tie=model.electrical(raw,scenario['fault_edge'])['feasible'],
      interpretation='Constructed semantic regression probe; not an empirical false-safe rate or validated relay study.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--output',default='results/stage1')
    args=parser.parse_args()
    out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
    case=json.loads((ROOT/'data/network.json').read_text())
    scenarios=json.loads((ROOT/'data/scenarios.json').read_text())
    if args.limit: scenarios=scenarios[:args.limit]
    policies=['no_restoration','fixed_rule','mw_first_rule','nominal_sequence_search']
    rows=[]; audit=[]; probes=[]; start=time.perf_counter()
    with (out/'run.log').open('w') as log, (out/'traces.jsonl').open('w') as trace, (out/'electrical_states.jsonl').open('w') as states:
        for scenario in scenarios:
            # A normal pre-fault power flow must pass before this contingency is accepted.
            pre_ac=ACModel(case,scenario)
            pre_edges=[e['id'] for e in case['edges'] if e['id']!='l3']
            pre_loads=[l['id'] for l in case['loads']]
            pre=pre_ac.solve(pre_edges,pre_loads)
            if not pre['feasible']: raise AssertionError(f'Invalid pre-fault case {scenario["id"]}: {pre["violations"]}')
            audit.append(dict(scenario=scenario['id'],state='pre_fault',**crosscheck(pre_ac.case,pre_edges,pre_loads,pre)))
            states.write(json.dumps(dict(scenario=scenario['id'],state='pre_fault',active_edges=pre_edges,enabled_loads=pre_loads,ac=pre))+'\n')
            verified=set()
            for policy in policies:
                ac=ACModel(case,scenario)
                env=Environment(ac,scenario)
                row=env.run(policy); row['ac_solver_calls']=ac.calls
                rows.append(row)
                trace.write(json.dumps(dict(scenario=scenario['id'],policy=policy,events=env.trace))+'\n')
                for edges,loads,pf in env.snapshots:
                    key=(tuple(edges),tuple(loads))
                    if key in verified: continue
                    audit.append(dict(scenario=scenario['id'],state=f'visited_{len(verified)}',
                                      **crosscheck(ac.case,edges,loads,pf)))
                    states.write(json.dumps(dict(scenario=scenario['id'],state=f'visited_{len(verified)}',active_edges=edges,enabled_loads=loads,ac=pf))+'\n')
                    verified.add(key)
            probes.append(history_probe(case,scenario))
            line=f'{scenario["id"]} {scenario["event_class"]}: '+', '.join(f'{r["policy"]}={r["ens_mwh"]:.6f} MWh' for r in rows[-4:])
            print(line,flush=True); log.write(line+'\n'); log.flush(); trace.flush()
            with (out/'metrics.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump(out/'independent_ac_checks.json',audit)
    dump(out/'history_probes.json',probes)
    summary={}
    for policy in policies:
        subset=[r for r in rows if r['policy']==policy]
        summary[policy]={name:sum(r[name] for r in subset)/len(subset) for name in
            ['ens_mwh','served_fraction','recovered_interrupted_fraction','commands','failed_commands','decision_wall_s','run_wall_s']}
        summary[policy]['full_restoration_cases']=sum(r['served_fraction']>1-1e-8 for r in subset)
        summary[policy]['unsafe_executed_states']=sum(r['unsafe_executed_states'] for r in subset)
    pairwise=[]
    for baseline in ['fixed_rule','mw_first_rule']:
        a={r['scenario']:r for r in rows if r['policy']==baseline}
        b={r['scenario']:r for r in rows if r['policy']=='nominal_sequence_search'}
        delta=[a[k]['ens_mwh']-b[k]['ens_mwh'] for k in a]
        pairwise.append(dict(baseline=baseline,mean_ens_reduction_mwh=sum(delta)/len(delta),
            improved=sum(x>1e-8 for x in delta),tied=sum(abs(x)<=1e-8 for x in delta),worse=sum(x< -1e-8 for x in delta),
            max_gain_mwh=max(delta),min_gain_mwh=min(delta)))
    summary['paired_comparisons']=pairwise
    summary['scenario_count']=len(scenarios)
    summary['independent_unique_states_checked']=len(audit)
    summary['max_voltage_error_pu']=max(a['max_voltage_error_pu'] for a in audit)
    summary['max_branch_power_error_mva']=max(a['max_branch_power_error_mva'] for a in audit)
    summary['max_current_error_ka']=max(a['max_current_error_ka'] for a in audit)
    summary['elapsed_wall_s']=time.perf_counter()-start
    summary['scope']='Frozen synthetic diagnostic cases; 1800 s ENS horizon. No RL trained; no field or statistical safety validation.'
    dump(out/'summary.json',summary)
    paths=sorted((ROOT/'src').glob('*.py'))+sorted((ROOT/'data').glob('*.json'))+sorted((ROOT/'tests').glob('*.py'))+[ROOT/'requirements-lock.txt',ROOT/'RUN_STAGE1.txt']
    manifest={'utc_time':datetime.now(timezone.utc).isoformat(),'argv':sys.argv,'python':sys.version,'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ['pandapower','numpy','scipy','pandas','networkx','matplotlib','pytest']},
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'seeds':[s['seed'] for s in scenarios],
        'timing':'Fresh AC model/cache per policy. Controller time includes primary AC checks, excludes independent audit and network construction.'}
    dump(out/'manifest.json',manifest)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
