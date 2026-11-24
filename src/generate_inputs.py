"""Generate a fully synthetic, anonymized stage-1 110 kV benchmark."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    buses = [{'id': n, 'vn_kv': v} for n, v in [('XA', 220), ('XB', 220), ('A', 110), ('B', 110)]]
    buses += [{'id': f'H{i}', 'vn_kv': 110} for i in range(6)]
    buses += [{'id': f'L{i}', 'vn_kv': 20} for i in range(6)]
    path = ['A'] + [f'H{i}' for i in range(6)] + ['B']
    edges = []
    for i, length in enumerate([7, 10, 8, 12, 9, 11, 6]):
        edges.append(dict(id=f'l{i}', kind='line', **{'from': path[i], 'to': path[i+1]},
                          length_km=length, r_ohm=0.194*length, x_ohm=0.41*length,
                          c_nf=8.75*length, max_i_ka=0.47))
    for name, a, b, sn, vkr, vk in [('tA','XA','A',100,0.26,12), ('tB','XB','B',100,0.26,12)] + [
            (f't{i}', f'H{i}', f'L{i}',25,0.41,12) for i in range(6)]:
        edges.append(dict(id=name, kind='trafo', **{'from': a, 'to': b}, sn_mva=sn,
                          vkr_percent=vkr, vk_percent=vk, pfe_kw=0, i0_percent=0, shift_degree=0))
    loads = [dict(id=f'd{i}', bus=f'L{i}', p_mw=p, q_mvar=p*np.tan(np.arccos(0.95)))
             for i,p in enumerate([7., 11., 9., 10., 8., 12.])]
    case = dict(name='SYNTHETIC_110KV_OPEN_RING_V1', synthetic=True, base_mva=100., frequency_hz=50.,
                buses=buses, edges=edges, loads=loads,
                sources=[dict(id='SA', bus='XA', vm_pu=1.02), dict(id='SB', bus='XB', vm_pu=1.02)],
                normally_open=['l3'], voltage_limits_pu=[0.95,1.05], loading_limit_percent=100.)
    provenance = {
      'classification':'Entire network, geometry, loads, timings and events are synthetic; no field data.',
      'parameter_source':'pandapower public standard-type documentation: 149-AL1/24-ST1A 110.0; 100 MVA 220/110 kV; 25 MVA 110/20 kV leakage parameters.',
      'url':'https://pandapower.readthedocs.io/en/latest/std_types/basic.html',
      'simplifications':['Balanced positive-sequence constant-PQ AC model.',
        'Transformer excitation losses, phase shifts and tap action set to zero; nominal voltage ratios retained.',
        'Source MW ceilings are synthetic operational budgets, distinct from transformer MVA ratings.',
        'Faulted branches removed from AC model; fault-contact and isolation hazards checked separately.',
        'No relay setting, electromagnetic transient or field protection certification is represented.',
        'Sequential commands; elapsed execution and acknowledgement time explicitly integrated.',
        'Historical private files inform abstract event categories only.'],
      'generation_seed':20261015,
      'synthetic_action_durations_s':{'isolate_endpoint':20,'confirm_missing_ack':60,'reset_protection':10,'close_root_or_tie':30,'pickup_load':10,'rearm_after_refusal':40,'wait_poll':5,'missing_ack_timeout':30},
      'timing_semantics':'A command changes the physical state at completion, then waits for its acknowledgement. Only then can a new command be selected; no overlapping actions. Controllers use a nominal 5 s acknowledgement for planning.',
      'scenario_design':'6 fault positions x 6 event classes x 2 independent load realizations = 72 fixed cases; diagnostic baseline, not a held-out RL test set.'}
    scenarios=[]
    event_types=['normal','slow_ack','tie_refusal','missing_isolation_ack','restricted_capacity','late_fault_report']
    for k,(fault,event,rep) in enumerate((f,e,r) for f in [0,1,2,4,5,6] for e in event_types for r in range(2)):
        seed=20261015+k
        rng=np.random.default_rng(seed)
        scale=rng.uniform(0.95,1.25,6).tolist()
        scenarios.append(dict(id=f'C{k:03d}',seed=seed,fault_edge=f'l{fault}',event_class=event,
            load_scale=scale,source_cap_mw=45. if event=='restricted_capacity' else 75.,
            fault_report_s=90. if event=='late_fault_report' else 5.,
            acknowledgement_s=40. if event=='slow_ack' else 5.,
            tie_refusal_once=event=='tie_refusal',missing_isolation_ack=event=='missing_isolation_ack',
            horizon_s=1800.,nominal_ack_s=5.))
    plan={
      'stage':'1: runnable AC and deterministic sequential-restoration baselines',
      'available':['5 historic operational source files for abstract event logic','3 prospective drafts','5 prior manuscripts'],
      'missing':['Validated complete 110 kV network R/X/B and equipment limits','Protection settings and validated action/interlock logic','SCADA/SOE timestamps and failure probabilities','Real load and topology histories'],
      'design':['Freeze synthetic inputs and seeds before comparisons','Run no-restoration, fixed-rule, MW-first rule and exhaustive nominal-sequence search','Check every executed electrical state using pandapower AC and independent Ybus solver','Exercise topology/protection/history edge cases with tests'],
      'metrics':['ENS_MWh over 1800 s','time_to_last_pickup_s','restored_load_fraction','commands and failed commands','unsafe_executions','AC verification error','wall clock time'],
      'next_stage_gate':'Train safe RL only after verified baselines, broader topologies, explicit observation API and meaningful runtime/scaling or restoration gains justify learning.',
      'split':'All 72 are frozen stage-1 diagnostics. Later learning uses disjoint topology/event seeds; these are not claimed as blind test data.'}
    for name,obj in [('network.json',case),('scenarios.json',scenarios),('provenance.json',provenance)]:
        (ROOT/'data'/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    (ROOT/'results'/'stage1_plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    print(f'Generated {len(buses)} buses, {len(edges)} branches, {len(scenarios)} scenarios; all synthetic.')

if __name__=='__main__':
    main()
