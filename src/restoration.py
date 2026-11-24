"""Sequential synthetic restoration; policy observations exclude future events.

This is a balanced steady-state research environment, not a relay/EMT model.
All compared policies use the same deterministic admissibility gate.
"""
from dataclasses import dataclass, replace, asdict
from functools import lru_cache
import time


@dataclass(frozen=True)
class State:
    reported: bool = False
    isolation: tuple = (0,0)  # 0 unconfirmed/closed; 1 confirmed open; 2 open but no acknowledgement
    protection_reset: bool = False
    root_closed: bool = False
    tie_closed: bool = False
    tie_locked: bool = False
    enabled: tuple = ()


@dataclass(frozen=True)
class Observation:
    state: State
    elapsed_s: float
    horizon_s: float
    fault_edge: str | None
    root_edge: str
    nominal_ack_s: float


def initial_state(fault,case=None):
    n=len(case['loads']) if case else 6
    mid=n//2
    side=range(mid) if int(fault[1:])<mid else range(mid,n)
    return State(enabled=tuple(f'd{i}' for i in range(n) if i not in side))


def root_edge(fault,case=None):
    n=len(case['loads']) if case else 6
    return 'l0' if int(fault[1:])<n//2 else f'l{n}'


def active_edges(case,state,fault):
    tie=case.get('normally_open',['l3'])[0]
    root=root_edge(fault,case)
    active={e['id'] for e in case['edges']} - {fault,tie,root}
    if state.root_closed and root!=fault:
        active.add(root)
    if state.tie_closed:
        active.add(tie)
    return sorted(active)


def nominal_transition(state,action):
    """Outcome used by policies: success and nominal acknowledgement, no oracle events."""
    if action.startswith('isolate_') or action.startswith('confirm_'):
        i=int(action[-1]); iso=list(state.isolation); iso[i]=1
        return replace(state,isolation=tuple(iso))
    if action=='reset_protection':
        return replace(state,protection_reset=True)
    if action=='close_root':
        return replace(state,root_closed=True)
    if action=='close_tie':
        return replace(state,tie_closed=True)
    if action=='rearm_tie':
        return replace(state,tie_locked=False)
    if action.startswith('pickup_'):
        return replace(state,enabled=tuple(sorted(state.enabled+(action[7:],))))
    return state


def physical_duration(action):
    if action.startswith('isolate_'): return 20.
    if action.startswith('confirm_'): return 60.
    if action.startswith('pickup_'): return 10.
    return {'reset_protection':10.,'close_root':30.,'close_tie':30.,'rearm_tie':40.,'wait':5.}[action]


class DecisionModel:
    """Receives public network, measured loads/budgets and the current observation only."""
    def __init__(self,ac):
        self.ac=ac
        self.total_mw=sum(l['p_mw'] for l in ac.case['loads'])

    def electrical(self,state,fault):
        return self.ac.solve(active_edges(self.ac.case,state,fault),state.enabled)

    def admissibility(self,obs,action):
        s=obs.state
        if action=='stop': return True, 'stop'
        if action=='wait': return not s.reported, 'waiting_for_fault_location'
        if not s.reported or obs.fault_edge is None: return False,'fault_location_unknown'
        if action.startswith('isolate_'):
            return s.isolation[int(action[-1])]==0, 'isolation_state'
        if action.startswith('confirm_'):
            return s.isolation[int(action[-1])]==2, 'acknowledgement_state'
        if action=='reset_protection':
            return s.isolation==(1,1) and not s.protection_reset,'isolation_acknowledgements_required'
        if action=='rearm_tie': return s.tie_locked,'refusal_latch'
        if action in ('close_root','close_tie'):
            if not s.protection_reset or s.isolation!=(1,1):
                return False,'protection_not_ready'
            if action=='close_root' and (s.root_closed or obs.root_edge==obs.fault_edge):
                return False,'root_unavailable'
            if action=='close_tie' and (s.tie_closed or s.tie_locked):
                return False,'tie_unavailable_or_locked'
        elif action.startswith('pickup_'):
            name=action[7:]
            if name not in {l['id'] for l in self.ac.case['loads']} or name in s.enabled:
                return False,'load_unavailable'
            live=self.electrical(s,obs.fault_edge)['live_buses']
            load=next(l for l in self.ac.case['loads'] if l['id']==name)
            if load['bus'] not in live: return False,'load_deenergized'
        else: return False,'unknown_action'
        result=self.electrical(nominal_transition(s,action),obs.fault_edge)
        return result['feasible'],','.join(result['violations']) or 'feasible'

    def actions(self,obs):
        names=['isolate_0','isolate_1','confirm_0','confirm_1','reset_protection',
               'close_root','close_tie','rearm_tie']+['pickup_'+l['id'] for l in self.ac.case['loads']]
        if not obs.state.reported: return ['wait']
        return [a for a in names if self.admissibility(obs,a)[0]]


def choose_rule(model,obs,mw_first=False):
    allowed=model.actions(obs)
    if not allowed: return 'stop'
    loads=['pickup_'+l['id'] for l in model.ac.case['loads']]
    if mw_first:
        loads=sorted(loads,key=lambda a:-next(l['p_mw'] for l in model.ac.case['loads'] if l['id']==a[7:]))
    order=['wait','isolate_0','isolate_1','confirm_0','confirm_1','reset_protection','close_root']+loads+['rearm_tie','close_tie']
    return next(a for a in order if a in allowed)


def choose_search(model,obs):
    """Exact finite monotone action enumeration under disclosed state/nominal durations.

    Replanned after every actual acknowledgement. Future refusal, report and delay
    realizations are absent from Observation and this routine.
    """
    if not obs.state.reported: return 'wait'
    fault=obs.fault_edge
    @lru_cache(None)
    def solve(s,remaining):
        served=model.electrical(s,fault)['served_mw']
        unserved=model.total_mw-served
        best=(unserved*remaining/3600.,0,'stop')
        view=replace(obs,state=s,elapsed_s=obs.horizon_s-remaining)
        for action in model.actions(view):
            duration=physical_duration(action)
            if duration>remaining: continue
            ack=min(obs.nominal_ack_s,remaining-duration)
            dt=duration+ack
            nxt=nominal_transition(s,action)
            after=model.total_mw-model.electrical(nxt,fault)['served_mw']
            if dt>=remaining:
                tail,number=0.,0
            else:
                tail,number,_=solve(nxt,round(remaining-dt,6))
            val=(unserved*duration/3600.+after*ack/3600.+tail,number+1,action)
            if val<best: best=val
        return best
    return solve(obs.state,round(obs.horizon_s-obs.elapsed_s,6))[2]


def physical_violations(case,state,fault,electrical):
    """Ground-truth check, independent of the policy permission predicate."""
    out=list(electrical.get('violations',[]))
    if not electrical.get('converged'): return out+['nonconverged']
    edge=next(e for e in case['edges'] if e['id']==fault)
    for i,node in enumerate((edge['from'],edge['to'])):
        contact_closed=state.isolation[i]==0
        # A feeder protection trip opens its source-side contact even for a root fault.
        if fault==root_edge(fault,case) and not state.root_closed and ((fault=='l0' and i==0) or (fault==f"l{len(case['loads'])}" and i==1)):
            contact_closed=False
        if contact_closed and node in electrical['live_buses']:
            out.append('energized_fault_contact_'+str(i))
    return out


class Environment:
    def __init__(self,ac,scenario):
        self.ac=ac; self.scenario=scenario; self.model=DecisionModel(ac)
        self.state=initial_state(scenario['fault_edge'],ac.case); self.t=0.; self.ens=0.
        self.trace=[]; self.snapshots=[]; self.failed=0; self.commands=0; self.unsafe=0
        self.refusal_used=False; self.missing_used=False; self.last_pickup=0.
        self.initial_unserved_mw=self.model.total_mw-self.result()['served_mw']
        self.record('initial')

    def observe(self):
        return Observation(self.state,self.t,self.scenario['horizon_s'],
                           self.scenario['fault_edge'] if self.state.reported else None,
                           root_edge(self.scenario['fault_edge'],self.ac.case),self.scenario['nominal_ack_s'])

    def result(self):
        return self.model.electrical(self.state,self.scenario['fault_edge'])

    def advance(self,seconds):
        dt=min(seconds,self.scenario['horizon_s']-self.t)
        self.ens+=(self.model.total_mw-self.result()['served_mw'])*dt/3600.
        self.t+=dt

    def record(self,event):
        r=self.result()
        violations=physical_violations(self.ac.case,self.state,self.scenario['fault_edge'],r)
        self.unsafe+=bool(violations)
        self.trace.append({'time_s':self.t,'event':event,'served_mw':r['served_mw'],
                           'ens_mwh':self.ens,'state':asdict(self.state),'violations':violations})
        self.snapshots.append((active_edges(self.ac.case,self.state,self.scenario['fault_edge']),list(self.state.enabled),r))
        if violations:
            raise RuntimeError(f'Physical violation in executed state: {violations}')

    def step(self,action):
        valid,reason=self.model.admissibility(self.observe(),action)
        if not valid: raise ValueError(f'Rejected action {action}: {reason}')
        if action=='stop':
            self.advance(self.scenario['horizon_s']-self.t); self.record('horizon'); return
        if action=='wait':
            self.advance(5.)
            if self.t>=self.scenario['fault_report_s']:
                self.state=replace(self.state,reported=True)
            self.record('wait/report'); return
        duration=physical_duration(action)
        self.commands+=1
        if self.t+duration>self.scenario['horizon_s']:
            self.advance(self.scenario['horizon_s']-self.t)
            self.record(action+':pending_physical_at_horizon'); return
        self.advance(duration)
        next_state=nominal_transition(self.state,action)
        outcome='success'
        if action=='close_tie' and self.scenario['tie_refusal_once'] and not self.refusal_used:
            next_state=replace(self.state,tie_locked=True); self.refusal_used=True; self.failed+=1; outcome='refused'
        if action.startswith('isolate_') and self.scenario['missing_isolation_ack'] and not self.missing_used:
            iso=list(next_state.isolation); iso[int(action[-1])]=2
            next_state=replace(next_state,isolation=tuple(iso)); self.missing_used=True; outcome='ack_missing'
        self.state=next_state
        if action.startswith('pickup_'): self.last_pickup=self.t
        self.record(action+':physical:'+outcome)
        ack=30. if outcome=='ack_missing' else self.scenario['acknowledgement_s']
        ack_due=self.t+ack
        self.advance(ack)
        stage='observed' if self.t>=ack_due else 'pending_ack_at_horizon'
        self.record(action+':'+stage+':'+outcome)

    def run(self,policy):
        t0=time.perf_counter(); decision_time=0.
        for _ in range(200):
            if self.t>=self.scenario['horizon_s']: break
            start=time.perf_counter()
            if policy=='no_restoration': action='stop'
            elif policy=='fixed_rule': action=choose_rule(self.model,self.observe())
            elif policy=='mw_first_rule': action=choose_rule(self.model,self.observe(),True)
            elif policy=='nominal_sequence_search': action=choose_search(self.model,self.observe())
            else: raise ValueError(policy)
            decision_time+=time.perf_counter()-start
            self.step(action)
        else: raise RuntimeError('step limit')
        r=self.result()
        return dict(scenario=self.scenario['id'],event_class=self.scenario['event_class'],seed=self.scenario['seed'],
            policy=policy,ens_mwh=self.ens,served_fraction=r['served_mw']/self.model.total_mw,
            recovered_interrupted_fraction=1-(self.model.total_mw-r['served_mw'])/self.initial_unserved_mw,
            full_restoration_time_s=self.last_pickup if abs(self.model.total_mw-r['served_mw'])<1e-8 else None,
            final_unserved_mw=self.model.total_mw-r['served_mw'],time_to_last_pickup_s=self.last_pickup,
            commands=self.commands,failed_commands=self.failed,unsafe_executed_states=self.unsafe,
            decision_wall_s=decision_time,run_wall_s=time.perf_counter()-t0)
