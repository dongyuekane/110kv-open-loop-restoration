"""Independent balanced AC power flow for the declared synthetic network.

No pandapower imports or conversions. Builds its own pi-branch Ybus and solves
constant-PQ nodal equations in rectangular voltages. One slack per energized
island, rated transformer ratio (unity on nominal voltage bases), zero phase
shift and excitation. Unsupplied islands are omitted from `bus`; active branches
in those islands have zero voltage/current/power. Unsupplied enabled loads are
listed explicitly. Unknown IDs and multi-source islands raise ValueError.
"""
import math
import numpy as np
from scipy.optimize import root


def _indexed(rows, kind):
    ans = {r['id']: r for r in rows}
    if len(ans) != len(rows):
        raise ValueError('Duplicate ' + kind + ' IDs')
    return ans


def solve_ac(case, active_edges, enabled_loads, available_sources):
    buses = _indexed(case['buses'], 'bus')
    edges = _indexed(case['edges'], 'edge')
    loads = _indexed(case['loads'], 'load')
    sources = _indexed(case['sources'], 'source')
    active_edges, enabled_loads, available_sources = map(list, (active_edges, enabled_loads, available_sources))
    for ids, mapping, label in [(active_edges, edges, 'edge'), (enabled_loads, loads, 'load'), (available_sources, sources, 'source')]:
        if len(set(ids)) != len(ids) or any(i not in mapping for i in ids):
            raise ValueError('Duplicate or unknown active ' + label + ' ID')
    base = float(case['base_mva']); hz = float(case.get('frequency_hz', 50.0))
    if base <= 0 or hz <= 0:
        raise ValueError('Base power and frequency must be positive')
    for obj in list(loads.values()) + list(sources.values()):
        if obj['bus'] not in buses:
            raise ValueError('Unknown load/source bus')
    adjacency = {k: set() for k in buses}
    branch_params = {}
    for eid in active_edges:
        e = edges[eid]; f, t = e['from'], e['to']
        if f not in buses or t not in buses or f == t:
            raise ValueError('Invalid edge terminals: ' + eid)
        vf, vt = float(buses[f]['vn_kv']), float(buses[t]['vn_kv'])
        if min(vf, vt) <= 0:
            raise ValueError('Nonpositive nominal voltage')
        if e['kind'] == 'line':
            if abs(vf-vt) > 1e-9:
                raise ValueError('Line connects unlike nominal voltage bases')
            zbase = vf*vf/base
            z = complex(float(e['r_ohm']), float(e['x_ohm']))/zbase
            bsh = 2*math.pi*hz*float(e.get('c_nf', 0))*1e-9*zbase
            yhalf = 0.5j*bsh
        elif e['kind'] == 'trafo':
            if any(abs(float(e.get(k, 0))) > 1e-12 for k in ['pfe_kw','i0_percent','shift_degree']):
                raise ValueError('Only zero-excitation, zero-phase-shift transformers supported')
            rr = float(e['vkr_percent']); zz = float(e['vk_percent']); sn = float(e['sn_mva'])
            if rr < 0 or zz <= abs(rr) or sn <= 0:
                raise ValueError('Invalid transformer impedance/nameplate')
            z = complex(rr, math.sqrt(zz*zz-rr*rr))/100*base/sn
            yhalf = 0j
        else:
            raise ValueError('Unknown edge kind: ' + str(e['kind']))
        if abs(z) <= 1e-14 or z.real < 0:
            raise ValueError('Invalid branch impedance: ' + eid)
        branch_params[eid] = (f, t, 1/z, yhalf)
        adjacency[f].add(t); adjacency[t].add(f)
    remaining = set(buses); components = []
    while remaining:
        start = next(iter(remaining)); stack = [start]; comp = set()
        while stack:
            k = stack.pop()
            if k in comp: continue
            comp.add(k); stack.extend(adjacency[k] - comp)
        remaining -= comp; components.append(comp)
    vcomplex = {}; bus_out = {}; source_p = {}; maxres = 0.; nfev = 0
    unsupplied = []; failures = []
    for comp in components:
        slack_ids = [s for s in available_sources if sources[s]['bus'] in comp]
        if len(slack_ids) > 1:
            raise ValueError('Multiple available sources in one island: ' + ','.join(sorted(slack_ids)))
        if not slack_ids:
            unsupplied.extend(k for k in enabled_loads if loads[k]['bus'] in comp)
            continue
        sid = slack_ids[0]; slack = sources[sid]['bus']; vm0 = float(sources[sid].get('vm_pu', 1.02))
        if not np.isfinite(vm0) or vm0 <= 0:
            raise ValueError('Invalid slack voltage')
        names = [slack] + sorted(comp-{slack}); idx = {n:i for i,n in enumerate(names)}
        n = len(names); Y = np.zeros((n,n),complex)
        for f,t,y,half in branch_params.values():
            if f not in comp: continue
            i,j = idx[f],idx[t]
            Y[i,i] += y+half; Y[j,j] += y+half; Y[i,j] -= y; Y[j,i] -= y
        specified = np.zeros(n,complex)
        for lid in enabled_loads:
            ld = loads[lid]
            if ld['bus'] in comp:
                specified[idx[ld['bus']]] -= complex(float(ld['p_mw']),float(ld['q_mvar']))/base
        pq = np.arange(1,n); G=Y.real; B=Y.imag; m=n-1
        def voltage(x):
            v=np.empty(n,complex); v[0]=vm0+0j
            v[1:]=x[:m]+1j*x[m:]
            return v
        def residual(x, fraction=1.):
            v=voltage(x); mis=v*np.conjugate(Y@v)-fraction*specified
            return np.r_[mis[1:].real,mis[1:].imag]
        def jacobian(x, fraction=1.):
            v=voltage(x); e=v.real; f=v.imag; cur=Y@v; ir=cur.real; ii=cur.imag
            pp=np.diag(ir)+e[:,None]*G+f[:,None]*B
            pf=np.diag(ii)-e[:,None]*B+f[:,None]*G
            qp=-np.diag(ii)+f[:,None]*G-e[:,None]*B
            qf=np.diag(ir)-f[:,None]*B-e[:,None]*G
            ix=np.ix_(pq,pq)
            return np.block([[pp[ix],pf[ix]],[qp[ix],qf[ix]]])
        if m:
            x0=np.r_[np.full(m,vm0),np.zeros(m)]
            sol=root(residual,x0,jac=jacobian,method='hybr',options={'xtol':1e-10,'maxfev':250})
            nfev+=sol.nfev; err=float(np.max(np.abs(residual(sol.x))))
            if not np.isfinite(err) or err>1e-9:
                # Independent continuation fallback follows the high-voltage
                # branch from small to full load, without altering the case.
                x=x0.copy()
                for fraction in (.1,.25,.5,.75,1.):
                    sol=root(residual,x,args=(fraction,),jac=jacobian,method='hybr',options={'xtol':1e-10,'maxfev':250})
                    nfev+=sol.nfev; x=sol.x
                    er=float(np.max(np.abs(residual(x,fraction))))
                    if not np.isfinite(er) or er>1e-9:break
                err=float(np.max(np.abs(residual(x))))
                v=voltage(x)
            else:v=voltage(sol.x)
            maxres=max(maxres,err)
            if not np.isfinite(err) or err>1e-9 or not np.all(np.isfinite(v)) or np.min(np.abs(v))<.05:
                failures.append({'island':names,'residual_pu':err if np.isfinite(err) else None})
                continue
        else:v=np.array([vm0+0j])
        injection=v*np.conjugate(Y@v)*base
        source_p[sid]=float(injection[0].real-specified[0].real*base)
        for k,value in zip(names,v):
            vcomplex[k]=value
            bus_out[k]={'vm_pu':float(abs(value)),'va_degree':float(np.angle(value,deg=True))}
    flows={}
    for eid,(f,t,y,half) in branch_params.items():
        vf=vcomplex.get(f,0j); vt=vcomplex.get(t,0j)
        iff=(vf-vt)*y+vf*half; itt=(vt-vf)*y+vt*half
        sf=vf*np.conjugate(iff)*base; st=vt*np.conjugate(itt)*base
        flows[eid]={'p_from_mw':float(sf.real),'q_from_mvar':float(sf.imag),'p_to_mw':float(st.real),'q_to_mvar':float(st.imag),
                    'i_from_ka':float(abs(iff)*base/(math.sqrt(3)*float(buses[f]['vn_kv']))),
                    'i_to_ka':float(abs(itt)*base/(math.sqrt(3)*float(buses[t]['vn_kv'])))}
    return {'converged':not failures,'bus':bus_out,'branches':flows,
            'loss_mw':float(sum(v['p_from_mw']+v['p_to_mw'] for v in flows.values())),
            'max_residual_pu':float(maxres) if np.isfinite(maxres) else None,
            'source_p_mw':source_p,'unsupplied_enabled_loads':sorted(unsupplied),'failed_islands':failures,'function_evaluations':nfev}
