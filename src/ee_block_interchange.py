"""Numerical verification of the conditional, non-interleaved block lemma."""
import json
from pathlib import Path


def committed_rollout(order, refused, root_mw=10., tie_mw=12.):
    enabled=set();t=0.;ens=0.;events=[]
    weights={'R':root_mw,'T':tie_mw}
    def interval(seconds, event, restore=None):
        nonlocal t,ens
        ens+=sum(v for k,v in weights.items() if k not in enabled)*seconds/3600.
        t+=seconds
        if restore is not None:
            enabled.add(restore)
        events.append(dict(time_s=t,event=event,ens_mwh=ens))
    for block in order:
        if block=='T' and refused:
            interval(30,'tie_close_refused');interval(5,'refusal_ack')
            interval(40,'rearm_complete');interval(5,'rearm_ack')
        interval(30,block+'_close_complete');interval(5,block+'_close_ack')
        interval(10,block+'_pickup_complete',restore=block);interval(5,block+'_pickup_ack')
    return dict(ens_mwh=ens,duration_s=t,events=events)


def main():
    rows=[]
    for p in (0.,.05,.1,.125,.25,.5,1.):
        values={}
        for order in ('RT','TR'):
            zero=committed_rollout(order,False)
            one=committed_rollout(order,True)
            values[order]=(1-p)*zero['ens_mwh']+p*one['ens_mwh']
        actual=values['RT']-values['TR']
        formula=(12*50-10*(50+p*80))/3600.
        assert abs(actual-formula)<1e-12
        rows.append(dict(p=p,rt_ens_mwh=values['RT'],tr_ens_mwh=values['TR'],
            difference_mwh=actual,formula_mwh=formula,error=abs(actual-formula)))
    result=dict(scope='Synthetic committed non-interleaved two-block verification',
        root_mw=10.,tie_mw=12.,normal_block_s=50.,refusal_extra_s=80.,threshold=.125,
        comparisons=rows,rollouts={o+str(b):committed_rollout(o,b) for o in ('RT','TR') for b in (False,True)})
    out=Path(__file__).resolve().parents[1]/'block_interchange_check.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':len(rows),'max_error_mwh':max(r['error'] for r in rows),'output':str(out)},indent=2))


if __name__=='__main__':main()
