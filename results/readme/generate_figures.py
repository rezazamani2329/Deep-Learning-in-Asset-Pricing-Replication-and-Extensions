from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
root=Path(__file__).resolve().parents[2]
t=root/'results/consolidated_comparison/tables'
f=root/'results/readme/figures'
models=['Official GAN','GAN–LSTM original','GAN–LSTM reduced schedule','GAN–Transformer reduced schedule','50/50 blend']
labels=['Authors’ GAN','Code 07 GAN','New LSTM','Transformer','50/50 blend']
colors=['#94a3b8','#003262','#2563eb','#b7790b','#16816b']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
bench=pd.read_csv(t/'all_models_vs_paper.csv')
blend=pd.read_csv(t/'architecture_blend_split_comparison.csv')
reg=pd.read_csv(t/'official_extended_regime_comparison.csv')
roll=pd.read_csv(t/'official_extended_rolling_series.csv',parse_dates=['date'])
summary=pd.read_csv(t/'official_extended_rolling_summary.csv')
end=pd.read_csv(t/'official_extended_endpoint_regimes.csv')
rows=[]
for name,src in [('GAN–LSTM (paper)','Paper GAN'),('GAN–LSTM original','Code 07 GAN'),('GAN–LSTM reduced schedule','New LSTM'),('GAN–Transformer reduced schedule','Transformer')]:
 r=bench.set_index('model').loc[name]
 rows.append([src,*[r[k] for k in ['train','validation','test']]])
rows.append(['50/50 blend',*[blend.query("model == '50/50 blend'").set_index('split').loc[k,'monthly_sharpe'] for k in ['train','validation','test']]])
fig,axs=plt.subplots(1,3,figsize=(14,4.6))
for j,(ax,split) in enumerate(zip(axs,['Training','Validation','Test'])):
 vals=[r[j+1] for r in rows];ax.bar(labels,vals,color=colors)
 ax.set_title(split);ax.set_ylabel('Monthly Sharpe');ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.2)
 for i,v in enumerate(vals):ax.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=10)
 ax.set_ylim(0,max(vals)*1.17)
fig.suptitle('Published GAN benchmark, replication, and reduced-budget extensions',fontsize=15)
fig.tight_layout();fig.savefig(f/'split_sharpe.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(13,8),sharex=True)
for ax,w in zip(axs,[36,60]):
 q=roll.query('window_months == @w')
 for model,label,col in zip(models,labels,colors):ax.plot(q.date,q[model],label=label,color=col,lw=1.6)
 ax.set_title(f'{w}-month complete rolling windows');ax.set_ylabel('Annualized rolling Sharpe');ax.grid(alpha=.2);ax.axhline(0,color='gray',lw=.6)
axs[0].legend(ncol=5,loc='upper right',fontsize=9)
fig.tight_layout();fig.savefig(f/'rolling_sharpe.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(14,5))
for ax,family,cats in zip(axs,['business_cycle','volatility'],[['Expansion','Recession'],['High volatility','Low volatility']]):
 for i,(model,label,col) in enumerate(zip(models,labels,colors)):
  vals=[reg.query('model == @model and regime == @cat').monthly_sharpe.iloc[0] for cat in cats]
  ax.bar(np.arange(2)+(i-2)*.15,vals,.15,label=label,color=col)
 ax.set_xticks(range(2),cats);ax.set_ylabel('Monthly conditional Sharpe');ax.set_title('Business cycle' if family=='business_cycle' else 'Lagged market volatility');ax.axhline(0,color='gray',lw=.7);ax.grid(axis='y',alpha=.2)
fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,labels)],loc='lower center',ncol=5);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(f/'regime_sharpe.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(14,5))
cats=['Expansion','Recession','High volatility','Low volatility']
for ax,w in zip(axs,[36,60]):
 for i,(model,label,col) in enumerate(zip(models,labels,colors)):
  vals=[end.query('window_months == @w and model == @model and endpoint_regime == @cat').mean_annualized_sharpe.iloc[0] for cat in cats]
  ax.bar(np.arange(4)+(i-2)*.15,vals,.15,color=col)
 ax.set_xticks(range(4),cats,rotation=20);ax.set_title(f'{w}-month windows grouped by ending-month regime');ax.set_ylabel('Mean annualized rolling Sharpe');ax.grid(axis='y',alpha=.2)
fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,labels)],loc='lower center',ncol=5);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(f/'rolling_endpoint_regimes.png',dpi=180);plt.close(fig)
