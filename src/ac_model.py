"""Balanced AC feasibility with explicit source/island and device constraints."""
from copy import deepcopy
import math
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'results'/'.mplconfig'))
import networkx as nx
import pandapower as pp


def components(case, active_edges, available_sources):
    graph=nx.Graph()
    graph.add_nodes_from(b['id'] for b in case['buses'])
    graph.add_edges_from((e['from'],e['to']) for e in case['edges'] if e['id'] in active_edges)
    sources={s['bus']:s['id'] for s in case['sources'] if s['id'] in available_sources}
    live=set()
    for comp in nx.connected_components(graph):
        count=sum(n in sources for n in comp)
        if count>1:
            raise ValueError('unintended_source_parallel')
        if count:
            if graph.subgraph(comp).number_of_edges()!=len(comp)-1:
                raise ValueError('energized_mesh')
            live.update(comp)
    return live


class ACModel:
    def __init__(self, case, scenario):
        self.case=deepcopy(case)
        for load,scale in zip(self.case['loads'],scenario['load_scale']):
            load['p_mw']*=scale
            load['q_mvar']*=scale
        self.source_cap_mw=scenario['source_cap_mw']
        self.cache={}
        self.calls=0
        self.net=pp.create_empty_network(sn_mva=case['base_mva'],f_hz=case['frequency_hz'])
        self.bus={b['id']:pp.create_bus(self.net,vn_kv=b['vn_kv'],name=b['id']) for b in case['buses']}
        self.edge={}
        for e in case['edges']:
            a,b=self.bus[e['from']],self.bus[e['to']]
            if e['kind']=='line':
                index=pp.create_line_from_parameters(self.net,a,b,length_km=e['length_km'],
                    r_ohm_per_km=e['r_ohm']/e['length_km'],x_ohm_per_km=e['x_ohm']/e['length_km'],
                    c_nf_per_km=e['c_nf']/e['length_km'],max_i_ka=e['max_i_ka'],name=e['id'])
            else:
                index=pp.create_transformer_from_parameters(self.net,a,b,sn_mva=e['sn_mva'],
                    vn_hv_kv=self.net.bus.at[a,'vn_kv'],vn_lv_kv=self.net.bus.at[b,'vn_kv'],
                    vkr_percent=e['vkr_percent'],vk_percent=e['vk_percent'],pfe_kw=e['pfe_kw'],
                    i0_percent=e['i0_percent'],shift_degree=e['shift_degree'],name=e['id'])
            self.edge[e['id']]=(e['kind'],index)
        self.load={l['id']:pp.create_load(self.net,self.bus[l['bus']],p_mw=l['p_mw'],q_mvar=l['q_mvar'],name=l['id']) for l in self.case['loads']}
        self.source={s['id']:pp.create_ext_grid(self.net,self.bus[s['bus']],vm_pu=s['vm_pu'],name=s['id']) for s in case['sources']}

    def solve(self, active_edges, enabled_loads, available_sources=('SA','SB')):
        key=(tuple(sorted(active_edges)),tuple(sorted(enabled_loads)),tuple(sorted(available_sources)))
        if key in self.cache:
            return self.cache[key]
        try:
            live=components(self.case,active_edges,available_sources)
        except ValueError as exc:
            return {'feasible':False,'converged':False,'violations':[str(exc)],'served_mw':0.}
        for name,i in self.bus.items():
            self.net.bus.at[i,'in_service']=name in live
        for name,(kind,i) in self.edge.items():
            self.net[kind].at[i,'in_service']=name in active_edges
        for name,i in self.load.items():
            self.net.load.at[i,'in_service']=name in enabled_loads
        for name,i in self.source.items():
            self.net.ext_grid.at[i,'in_service']=name in available_sources
        self.calls+=1
        try:
            pp.runpp(self.net,algorithm='nr',calculate_voltage_angles=True,init='flat',
                     tolerance_mva=1e-9,max_iteration=30,trafo_model='pi',numba=False)
        except pp.LoadflowNotConverged:
            result={'feasible':False,'converged':False,'violations':['ac_nonconvergence'],'served_mw':0.}
            self.cache[key]=result
            return result
        volt={n:{'vm_pu':float(self.net.res_bus.at[i,'vm_pu']),
                 'va_degree':float(self.net.res_bus.at[i,'va_degree'])} for n,i in self.bus.items() if n in live}
        flows={}
        for e in self.case['edges']:
            name=e['id']
            if name not in active_edges or e['from'] not in live:
                continue
            kind,i=self.edge[name]
            row=self.net['res_'+kind].loc[i]
            if kind=='line':
                vals={k:float(row[k]) for k in ['p_from_mw','q_from_mvar','p_to_mw','q_to_mvar','i_from_ka','i_to_ka']}
            else:
                vals={new:float(row[old]) for new,old in [('p_from_mw','p_hv_mw'),('q_from_mvar','q_hv_mvar'),('p_to_mw','p_lv_mw'),('q_to_mvar','q_lv_mvar'),('i_from_ka','i_hv_ka'),('i_to_ka','i_lv_ka')]}
            vals['loading_percent']=float(row['loading_percent'])
            flows[name]=vals
        injections={n:float(self.net.res_ext_grid.at[i,'p_mw']) for n,i in self.source.items() if n in available_sources}
        violations=[]
        vmin,vmax=self.case['voltage_limits_pu']
        for n,v in volt.items():
            if not vmin-1e-8<=v['vm_pu']<=vmax+1e-8:
                violations.append('voltage:'+n)
        for n,f in flows.items():
            if f['loading_percent']>100+1e-7:
                violations.append('loading:'+n)
        for n,p in injections.items():
            if p>self.source_cap_mw+1e-7:
                violations.append('source_budget:'+n)
        served=sum(l['p_mw'] for l in self.case['loads'] if l['id'] in enabled_loads and l['bus'] in live)
        result={'converged':True,'feasible':not violations,'violations':violations,
                'bus':volt,'branches':flows,'source_p_mw':injections,'served_mw':served,
                'loss_mw':sum(f['p_from_mw']+f['p_to_mw'] for f in flows.values()),
                'live_buses':sorted(live)}
        self.cache[key]=result
        return result
