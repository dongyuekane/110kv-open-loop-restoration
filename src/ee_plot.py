"""Publication figures from executed, paired simulation outputs."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ee/figures';OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.mplconfig'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7,'lines.linewidth':1.4})
COL=['#000000','#666666','#235d85','#b14d24']
LABEL={'fixed_rule':'Fixed rule','mw_first_rule':'MW-first rule','nominal_sequence_search':'Nominal search','contingent_0.25':'Contingent (p = 0.25)'}
POL=list(LABEL)
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=500,bbox_inches='tight',facecolor='white');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.eps'),bbox_inches='tight');plt.close(fig)

def network():
 fig=plt.figure(figsize=(6.85,2.8));ax=fig.add_axes([.03,.32,.94,.64]);ax.set_xlim(-.9,7.9);ax.set_ylim(-1.4,1.8);ax.axis('off')
 for i in range(7):
  if i==3:
   ax.plot([i,i+.35],[0,0],c='k');ax.plot([i+.65,i+1],[0,0],c='k');ax.plot([i+.35,i+.65],[0,.28],c='k');ax.text(i+.5,.95,'Normally open tie',ha='center',fontsize=8)
  else:ax.plot([i,i+1],[0,0],c='k')
  if i!=3:ax.text(i+.5,.22,f'l{i}',ha='center',fontsize=8)
 for i in [0,7]:
  ax.plot([i,i],[0,1],c='k');ax.plot(i,0,'ks',ms=5);ax.text(i,-.27,'A' if i==0 else 'B',ha='center');ax.text(i,1.45,'220 kV source',ha='center',fontsize=8);ax.text(i,1.12,'220/110 kV',ha='center',fontsize=8)
 for i in range(1,7):
  ax.plot(i,0,'ko',ms=4);ax.text(i,.45,f'H{i-1}',ha='center',fontsize=8);ax.plot([i,i],[0,-.55],c='k');ax.plot(i,-.6,'kv',ms=5);ax.text(i,-.95,f'd{i-1}',ha='center',fontsize=8)
 ax.text(3.5,-1.32,'110/20 kV station transformers and aggregate PQ loads',ha='center',fontsize=8)
 ax2=fig.add_axes([.03,.03,.94,.25]);ax2.axis('off');ax2.set_xlim(0,10);ax2.set_ylim(0,1)
 texts=['Observed state','Readiness + AC\nchecks','Contingent\nsequence search','Physical\ncompletion','Acknowledgement\n+ new decision']
 for i,t in enumerate(texts):
  x=i*2+.02;ax2.add_patch(Rectangle((x,.25),1.8,.65,facecolor='#eeeeee',edgecolor='0.4',linewidth=.6));ax2.text(x+.9,.57,t,ha='center',va='center',fontsize=8)
  if i<4:ax2.annotate('',xy=(x+1.99,.56),xytext=(x+1.8,.56),arrowprops=dict(arrowstyle='->',lw=.7))
 save(fig,'Fig1')

def core_figs():
 means=pd.read_csv(ROOT/'results/ee/core/expected_summary.csv');cases=pd.read_csv(ROOT/'results/ee/core/expected_by_case.csv')
 fig,axs=plt.subplots(1,2,figsize=(6.85,3.2));fig.subplots_adjust(wspace=.4,bottom=.30,top=.9)
 for i,pol in enumerate(POL):
  d=means[means.policy==pol];axs[0].plot(d.p_actual,d.ens_mwh,marker=['o','s','^','D'][i],ms=3,color=COL[i],label=LABEL[pol])
 axs[0].set_xlabel('Evaluation refusal probability');axs[0].set_ylabel('Expected ENS (MWh)');axs[0].text(.0,1.05,'a',transform=axs[0].transAxes,fontweight='bold');fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.3,-.015),ncol=2,fontsize=7,frameon=False)
 nominal=means[means.policy=='nominal_sequence_search'].set_index('p_actual').ens_mwh
 curves=[]
 for p in [0.,.05,.1,.25,.5]:
  pol='nominal_sequence_search' if p==0 else f'contingent_{p:.2f}';curves.append((p,(nominal[p]-means[(means.policy==pol)&(means.p_actual==p)].iloc[0].ens_mwh)*1000))
 axs[1].plot(*zip(*curves),'o-',color=COL[2],ms=4,label='Matched planning probability')
 d=means[means.policy=='contingent_0.25'].set_index('p_actual');axs[1].plot(d.index,(nominal-d.ens_mwh)*1000,'s--',color=COL[3],ms=4,label='Fixed planning probability 0.25')
 axs[1].axhline(0,c='0.5',lw=.6);axs[1].set_xlabel('Evaluation refusal probability');axs[1].set_ylabel('ENS reduction versus nominal (kWh)');axs[1].text(.0,1.05,'b',transform=axs[1].transAxes,fontweight='bold');axs[1].legend(fontsize=7,frameon=False)
 save(fig,'Fig2')
 # Mechanism illustration selected by declared largest incremental expected gain.
 z=cases[cases.p_actual==.25].pivot(index='base_case',columns='policy',values='ens_mwh');diff=z.nominal_sequence_search-z['contingent_0.25'];example=str(diff.idxmax())
 records=[json.loads(line) for line in (ROOT/'results/ee/core/traces.jsonl').read_text().splitlines()]
 target=[r for r in records if r['base_case']==example and r['policy'] in ['nominal_sequence_search','contingent_0.25']]
 fig,axs=plt.subplots(1,2,figsize=(6.85,2.6),sharey=True);fig.subplots_adjust(wspace=.15,bottom=.22,top=.83)
 timeline=[]
 for outcome,ax in enumerate(axs):
  for i,pol in enumerate(['nominal_sequence_search','contingent_0.25']):
   r=next(r for r in target if r['policy']==pol and r['scenario'].endswith(f'_Y{outcome}'));e=r['events'];initial=e[0]['served_mw'];x=[v['time_s'] for v in e];y=[v['served_mw']-initial for v in e]
   ax.step(x,y,where='post',color=COL[i+2],linestyle=['--','-'][i],label=LABEL[pol]);timeline.append(r)
  ax.set_xlim(0,400);ax.set_xlabel('Time after fault (s)');ax.text(.02,1.04,('a  First closure succeeds' if outcome==0 else 'b  First closure is refused'),transform=ax.transAxes,va='bottom',fontsize=8)
 axs[0].set_ylabel('Recovered interrupted demand (MW)');axs[1].legend(loc='lower right',frameon=False,fontsize=7)
 save(fig,'Fig3')
 (OUT/'figure_selection.json').write_text(json.dumps({'example':example,'criterion':'Largest paired expected ENS reduction of contingent p=.25 against nominal at evaluation p=.25 within all144corecases. Mechanism illustration, not average performance.','expected_gain_kwh':float(diff.max()*1000),'records':timeline},indent=2))
 # Cross-probability regret surface.
 planps=[0.,.05,.1,.25,.5];mat=np.zeros((5,5))
 for j,p in enumerate(planps):
  best=means[means.p_actual==p].ens_mwh.min()
  for i,q in enumerate(planps):
   pol='nominal_sequence_search' if q==0 else f'contingent_{q:.2f}';mat[i,j]=(means[(means.p_actual==p)&(means.policy==pol)].iloc[0].ens_mwh-best)*1000
 fig,ax=plt.subplots(figsize=(4.3,3));im=ax.imshow(mat,cmap='Greys',origin='lower',aspect='auto');ax.set_xticks(range(5),[f'{p:g}' for p in planps]);ax.set_yticks(range(5),[f'{p:g}' for p in planps]);ax.set_xlabel('Evaluation refusal probability');ax.set_ylabel('Planning refusal probability')
 for i in range(5):
  for j in range(5):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]>mat.max()*.55 else 'black',fontsize=9)
 cb=fig.colorbar(im,ax=ax);cb.set_label('Excess expected ENS (kWh)');fig.tight_layout();save(fig,'Fig4')
 (OUT/'probability_regret.json').write_text(json.dumps({'planning_probabilities':planps,'evaluation_probabilities':planps,'excess_expected_ens_kwh':mat.tolist()},indent=2))

if __name__=='__main__':network();core_figs()
