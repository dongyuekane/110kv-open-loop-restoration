"""Paired, exact-outcome evaluation of the frozen EE protocol."""
import argparse,csv,hashlib,importlib.metadata,json,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
from copy import deepcopy
from ac_model import ACModel
from restoration import Environment
from contingent_planner import run_contingent
from run_stage1 import crosscheck
ROOT=Path(__file__).resolve().parents[1]
POLICIES=['fixed_rule','mw_first_rule','nominal_sequence_search','contingent_0.05','contingent_0.10','contingent_0.25','contingent_0.50']
def main():
 p=argparse.ArgumentParser();p.add_argument('--suite',choices=['core','stress'],default='core');p.add_argument('--limit',type=int,default=0);p.add_argument('--output');args=p.parse_args()
 out=ROOT/(args.output or 'results/ee/'+args.suite);out.mkdir(parents=True,exist_ok=True)
 cases=json.loads((ROOT/f'data/ee/{args.suite}_cases.json').read_text());nets=json.loads((ROOT/'data/ee/networks.json').read_text())
 if args.limit:cases=cases[:args.limit]
 rows=[];audits=[];t0=time.perf_counter()
 with (out/'traces.jsonl').open('w') as tr,(out/'electrical_states.jsonl').open('w') as ef,(out/'run.log').open('w') as log:
  for base in cases:
   net=nets[str(base['n_stations'])];verified=set();count0=len(rows)
   ac=ACModel(net,base);edges=[e['id'] for e in net['edges'] if e['id'] not in net['normally_open']];loads=[l['id'] for l in net['loads']];pre=ac.solve(edges,loads)
   if not pre['feasible']:raise AssertionError((base['id'],'prefault',pre['violations']))
   audits.append(dict(base_case=base['id'],state='prefault',**crosscheck(ac.case,edges,loads,pre)))
   for refused in [False,True]:
    sc=deepcopy(base);sc['id']+=f'_Y{int(refused)}';sc['tie_refusal_once']=refused
    for name in POLICIES:
     ac=ACModel(net,sc);env=Environment(ac,sc)
     if name.startswith('contingent_'):row,plans=run_contingent(env,float(name.split('_')[1]))
     else:row=env.run(name);plans=[]
     row.update(policy=name,base_case=base['id'],refusal_outcome=int(refused),n_stations=base['n_stations'],source_cap_mw=base['source_cap_mw'],fault_edge=base['fault_edge'],load_seed=base['seed'],ac_solver_calls=ac.calls)
     if 'assumed_refusal_probability' not in row:row['assumed_refusal_probability']=0. if name=='nominal_sequence_search' else None
     row.setdefault('decision_calls',None);row.setdefault('max_planning_states',None)
     rows.append(row)
     tr.write(json.dumps(dict(base_case=base['id'],scenario=sc['id'],policy=name,events=env.trace,plans=plans),allow_nan=False)+'\n')
     for edges,loads,pf in env.snapshots:
      key=(tuple(edges),tuple(loads))
      if key in verified:continue
      audits.append(dict(base_case=base['id'],state=f'visited_{len(verified)}',**crosscheck(ac.case,edges,loads,pf)))
      ef.write(json.dumps(dict(base_case=base['id'],active_edges=edges,enabled_loads=loads,ac=pf),allow_nan=False)+'\n')
      verified.add(key)
   line=f"{base['id']}: {len(rows)-count0} episodes, {len(verified)} unique visited AC states; elapsed {time.perf_counter()-t0:.1f} s"
   print(line,flush=True);log.write(line+'\n');log.flush();tr.flush();ef.flush()
   fields=sorted({k for r in rows for k in r})
   with (out/'metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 (out/'independent_ac_checks.json').write_text(json.dumps(audits,indent=2))
 paths=list((ROOT/'src').glob('*.py'))+list((ROOT/'data/ee').glob('*.json'))+[ROOT/'requirements-lock.txt']
 manifest=dict(utc_time=datetime.now(timezone.utc).isoformat(),argv=sys.argv,elapsed_wall_s=time.perf_counter()-t0,episodes=len(rows),base_cases=len(cases),python=sys.version,platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ['pandapower','numpy','scipy','pandas','networkx']},seeds=sorted({c['seed'] for c in cases}),sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},timing='Fresh AC model/cache for every policy and outcome; decision_wall_s excludes construction, independent AC audit, and serialization.')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ['episodes','base_cases','elapsed_wall_s']}))
if __name__=='__main__':main()
