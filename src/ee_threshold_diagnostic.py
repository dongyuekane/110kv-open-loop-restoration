"""Exploratory causal action-threshold diagnostic on preselected ring states.

This is a mechanism diagnostic, not an additional performance benchmark or a
runtime comparison. Electrical caches are reused within each background.
The paired real rollouts preserve contingent action interleaving.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from ac_model import ACModel
from restoration import Environment
from contingent_planner import ContingentPlanner, run_contingent
from run_stage1 import crosscheck

ROOT=Path(__file__).resolve().parents[1]
GRID=(0.,.05,.1,.125,.15,.2,.25,.3,.4,.5,.75,1.)
IDS=('N6_R0_C75_F1','N6_R0_C75_F2','N8_R0_C75_F1',
     'N8_R0_C75_F3','N10_R0_C75_F1','N10_R0_C75_F4')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data/ee')
    parser.add_argument('--output',type=Path,default=ROOT/'threshold_diagnostic.json')
    args=parser.parse_args()
    networks=json.loads((args.data/'networks.json').read_text())
    scenarios={s['id']:s for s in json.loads((args.data/'core_cases.json').read_text())}
    protocol=dict(selected_ids=IDS,probability_grid=GRID,
        selection='First root-generated load profile and high source capacity; fault immediately after first station and before central tie, for each size. Chosen before diagnostic outputs.',
        initial_state='Common observed fault report, two confirmed isolation actions, and protection reset completed with nominal acknowledgements.',
        scope='Exploratory mechanism scan; no performance-case selection. Each p is assumed and evaluated at the same value. Cache reuse excludes runtime claims.')
    protocol_path=args.output.with_name(args.output.stem+'_protocol.json')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    results=[];audits=[]
    for ident in IDS:
        background=scenarios[ident];case=networks[str(background['n_stations'])]
        ac=ACModel(case,background);seen=set();items=[];obs_state=None
        for p in GRID:
            branches=[];expected=None;first_action=None
            for refused in (False,True):
                cfg=dict(deepcopy(background),tie_refusal_once=refused)
                env=Environment(ac,cfg)
                while not env.state.reported:env.step('wait')
                for action in ('isolate_0','isolate_1','reset_protection'):env.step(action)
                obs=env.observe();obs_state=asdict(obs)
                initial=ContingentPlanner(env.model,p).plan(obs)
                if expected is None:
                    expected=env.ens+initial.expected_ens_mwh
                    first_action=initial.action
                row,plans=run_contingent(env,p)
                branches.append(dict(refusal=refused,ens_mwh=row['ens_mwh'],
                    actions=plans,physical_events=[e for e in env.trace if ':physical:' in e['event']],
                    unsafe_executed_states=row['unsafe_executed_states']))
                for edges,loads,result in env.snapshots:
                    key=(tuple(edges),tuple(loads))
                    if key not in seen:
                        audits.append(dict(background_id=ident,**crosscheck(ac.case,edges,loads,result)))
                        seen.add(key)
            actual=(1-p)*branches[0]['ens_mwh']+p*branches[1]['ens_mwh']
            assert abs(actual-expected)<1e-9
            items.append(dict(p=p,first_action=first_action,weighted_ens_mwh=actual,
                predicted_ens_mwh=expected,expectation_error=abs(actual-expected),branches=branches))
        intervals=[]
        for item in items:
            if not intervals or intervals[-1]['first_action']!=item['first_action']:
                intervals.append(dict(first_action=item['first_action'],grid_min_p=item['p'],grid_max_p=item['p']))
            else:intervals[-1]['grid_max_p']=item['p']
        result=dict(background_id=ident,decision_observation=obs_state,
            first_action_grid_intervals=intervals,scan=items)
        results.append(result)
        print(json.dumps({'background':ident,'action_intervals':intervals}),flush=True)
    output=dict(protocol=protocol,results=results,independent_ac_checks=audits,
        max_expectation_error=max(i['expectation_error'] for r in results for i in r['scan']))
    args.output.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
