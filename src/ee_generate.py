"""Freeze synthetic multi-size inputs before running any EE comparisons."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/ee'

def make_case(n):
    original=json.loads((ROOT/'data/network.json').read_text())
    c=deepcopy(original)
    c['name']=f'SYNTHETIC_110KV_RING_{n}_STATIONS'
    c['buses']=original['buses'][:4]+[{'id':f'H{i}','vn_kv':110} for i in range(n)]+[{'id':f'L{i}','vn_kv':20} for i in range(n)]
    path=['A']+[f'H{i}' for i in range(n)]+['B']
    lengths=[7,10,8,12,9,11,6]
    c['edges']=[]
    for i,(a,b) in enumerate(zip(path[:-1],path[1:])):
        km=lengths[i%len(lengths)]*6/n
        c['edges'].append(dict(id=f'l{i}',kind='line',**{'from':a,'to':b},length_km=km,r_ohm=.194*km,x_ohm=.41*km,c_nf=8.75*km,max_i_ka=.47))
    c['edges']+=deepcopy([e for e in original['edges'] if e['id'] in ['tA','tB']])
    proto=next(e for e in original['edges'] if e['id']=='t0')
    for i in range(n):
        c['edges'].append(dict(deepcopy(proto),id=f't{i}',**{'from':f'H{i}','to':f'L{i}'}))
    values=np.array([7,11,9,10,8,12]*(1+n//6),dtype=float)[:n]
    values=values*57/values.sum()
    c['loads']=[dict(id=f'd{i}',bus=f'L{i}',p_mw=float(p),q_mvar=float(p*np.tan(np.arccos(.95)))) for i,p in enumerate(values)]
    c['normally_open']=[f'l{n//2}']
    return c

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases={str(n):make_case(n) for n in [6,8,10]}
    configs=[]
    for n in [6,8,10]:
        for rep in range(3):
            seed=2026101500+100*n+rep
            scales=np.random.default_rng(seed).uniform(.85,1.25,n).tolist()
            for cap in [45.,75.]:
                for f in range(n+1):
                    if f==n//2:continue
                    key=f'N{n}_R{rep}_C{int(cap)}_F{f}'
                    configs.append(dict(id=key,n_stations=n,seed=seed,load_scale=scales,source_cap_mw=cap,fault_edge=f'l{f}',horizon_s=1800.,fault_report_s=5.,acknowledgement_s=5.,nominal_ack_s=5.,tie_refusal_once=False,missing_isolation_ack=False,event_class='paired_refusal_core'))
    stress=[]
    for c in configs:
        if '_R0_' not in c['id'] or c['source_cap_mw']!=75:continue
        for event in ['slow_ack','missing_isolation_ack','late_report']:
            x=deepcopy(c);x['id']+='_'+event;x['event_class']=event
            if event=='slow_ack':x['acknowledgement_s']=40.
            elif event=='missing_isolation_ack':x['missing_isolation_ack']=True
            else:x['fault_report_s']=90.
            stress.append(x)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),synthetic=True,
      scope='Balanced 110 kV open-ring variants with 6, 8, and 10 stations. Different ring sizes, not arbitrary-topology generalization.',
      core_base_cases=len(configs),stress_base_cases=len(stress),paired_outcomes=[False,True],assumed_refusal_probabilities=[0.,.05,.1,.25,.5],
      evaluation_refusal_probabilities=[0.,.05,.1,.25,.5],primary_evaluation_probability=.25,
      policies=['fixed_rule','mw_first_rule','nominal_sequence_search','contingent_0.05','contingent_0.10','contingent_0.25','contingent_0.50'],
      primary_estimand='Equal-weight mean over all base cases of paired exact expected ENS difference at p=0.25; no stochastic failure sampling.',
      uncertainty='Primary is a fully specified finite synthetic grid. Descriptive spread across cases is reported; no population significance claims. A seed-block bootstrap, if added, concerns only the specified synthetic load distribution.',
      failure_semantics='At most one refusal on the first attempted tie close, after which rearm and the next close succeed. No event realization is supplied to policies.',
      secondary='Assumed-vs-evaluation p matrix; all same-input paired comparisons; timing stresses retained even if no benefit.',
      pre_fault_rule='Every case must pass pre-fault AC. Any failure halts the run and requires a recorded protocol revision; no silent filtering.',
      data_origin='All geometry, loads, timings, failures and probabilities are synthetic. Standard-type electrical parameters are inherited from the disclosed stage1 input.')
    for name,data in [('networks.json',cases),('core_cases.json',configs),('stress_cases.json',stress),('protocol.json',protocol)]:
        (OUT/name).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(protocol,indent=2))
if __name__=='__main__':main()
