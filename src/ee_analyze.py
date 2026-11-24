"""Compute paired finite-grid summaries, without inferred field probabilities."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/ee'
PROBS=[0.,.05,.1,.25,.5]

def paired_summary(frame):
    stats=[];expected=[]
    for p in PROBS:
        for (case,pol),x in frame.groupby(['base_case','policy'],sort=True):
            x=x.set_index('refusal_outcome');r=x.loc[0]
            y=dict(base_case=case,policy=pol,p_actual=p,n_stations=int(r.n_stations),source_cap_mw=float(r.source_cap_mw),seed=int(r.seed),fault_edge=r.fault_edge,event_class=r.event_class)
            for k in ['ens_mwh','served_fraction','recovered_interrupted_fraction','final_unserved_mw','commands','failed_commands','decision_wall_s']:
                y[k]=(1-p)*float(x.loc[0,k])+p*float(x.loc[1,k])
            expected.append(y)
    e=pd.DataFrame(expected)
    for (p,pol),x in e.groupby(['p_actual','policy']):
        stats.append(dict(p_actual=p,policy=pol,base_cases=len(x),**{k:float(x[k].mean()) for k in ['ens_mwh','served_fraction','recovered_interrupted_fraction','final_unserved_mw','commands','failed_commands','decision_wall_s']}))
    return e,pd.DataFrame(stats)

def main():
    summaries={}
    for suite in ['core','stress']:
        path=OUT/suite/'metrics.csv'
        if not path.exists() or not (OUT/suite/'manifest.json').exists():continue
        df=pd.read_csv(path)
        expected,means=paired_summary(df)
        expected.to_csv(OUT/suite/'expected_by_case.csv',index=False);means.to_csv(OUT/suite/'expected_summary.csv',index=False)
        comparisons=[]
        for p in PROBS:
            z=expected[expected.p_actual==p].pivot(index='base_case',columns='policy',values='ens_mwh')
            for method in [c for c in z if c.startswith('contingent_')]:
                for base in ['fixed_rule','mw_first_rule','nominal_sequence_search']:
                    d=(z[base]-z[method])*1000
                    comparisons.append(dict(p_actual=p,method=method,baseline=base,mean_gain_kwh=float(d.mean()),total_gain_kwh=float(d.sum()),median_gain_kwh=float(d.median()),max_gain_kwh=float(d.max()),min_gain_kwh=float(d.min()),improved=int((d>1e-5).sum()),tied=int((abs(d)<=1e-5).sum()),worse=int((d< -1e-5).sum()),relative_ens_reduction_percent=float(100*(z[base].mean()-z[method].mean())/z[base].mean())))
        pd.DataFrame(comparisons).to_csv(OUT/suite/'paired_comparisons.csv',index=False)
        checks=json.loads((OUT/suite/'independent_ac_checks.json').read_text())
        summaries[suite]=dict(episodes=len(df),base_cases=int(df.base_case.nunique()),unsafe_executed_states=int(df.unsafe_executed_states.sum()),independent_states=len(checks),max_voltage_difference_pu=max(c['max_voltage_error_pu'] for c in checks),max_power_difference_mva=max(c['max_branch_power_error_mva'] for c in checks),primary_means=means[means.p_actual==.25].to_dict('records'),primary_comparisons=[c for c in comparisons if c['p_actual']==.25 and c['method']=='contingent_0.25'],timing={p:dict(mean_s=float(x.decision_wall_s.mean()),median_s=float(x.decision_wall_s.median()),max_s=float(x.decision_wall_s.max())) for p,x in df.groupby('policy')})
        bysize=expected[expected.p_actual==.25].groupby(['n_stations','policy'])[['ens_mwh','served_fraction','decision_wall_s']].mean().reset_index();bysize.to_csv(OUT/suite/'by_size.csv',index=False)
        byevent=expected[expected.p_actual==.25].groupby(['event_class','policy'])[['ens_mwh','served_fraction','decision_wall_s']].mean().reset_index();byevent.to_csv(OUT/suite/'by_event.csv',index=False)
        # Directly reconstruct ENS from physical load restoration times and terminal deficit.
        errors=[];trajectory=[]
        for line in (OUT/suite/'traces.jsonl').read_text().splitlines():
            record=json.loads(line);events=record['events'];last=events[-1];h=last['time_s'];initial=events[0]['served_mw'];final=last['served_mw']
            area=sum((events[i+1]['time_s']-a['time_s'])*(final-a['served_mw'])/3600 for i,a in enumerate(events[:-1]))
            row=df[(df.base_case==record['base_case'])&(df.policy==record['policy'])&(df.scenario==record['scenario'])].iloc[0]
            residual=float(row.final_unserved_mw)*h/3600
            errors.append(abs(float(row.ens_mwh)-residual-area))
            trajectory.append(dict(base_case=record['base_case'],scenario=record['scenario'],policy=record['policy'],persistent_deficit_ens_mwh=residual,restoration_delay_ens_mwh=area))
        pd.DataFrame(trajectory).to_csv(OUT/suite/'ens_decomposition.csv',index=False)
        summaries[suite]['max_ens_decomposition_error_mwh']=max(errors)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
