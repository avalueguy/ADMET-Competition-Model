"""Scientific descriptive plots from completed development evidence; no fitting."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
TITLES={'caco2_wang':'Caco2','ld50_zhu':'LD50','lipophilicity_astrazeneca':'Lipophilicity',
        'ppbr_az':'PPBR','solubility_aqsoldb':'Solubility','clearance_hepatocyte_az':'Hepatocyte clearance*',
        'clearance_microsome_az':'Microsome clearance*','half_life_obach':'Half-life*','vdss_lombardo':'VDss*'}
LABELS={'native_d3':'Median tree depth 3','native_d6':'Median tree depth 6','asinh_d3':'Asinh tree depth 3',
        'feature_transformer':'Transformer','retrieval_transformer':'Retrieval Transformer'}
COLORS=dict(zip(LABELS,('#3274a1','#59a14f','#f28e2b','#9467bd','#e15759')))


def main():
    records=[]
    for v in (377,378):
        records+=json.loads((ROOT/f'docs/v{v}_learning_curves.json').read_text())['records']
    fig,axes=plt.subplots(3,3,figsize=(13,11))
    fig.subplots_adjust(left=.075,right=.99,top=.925,bottom=.19,hspace=.36,wspace=.25)
    for ax,(task,title) in zip(axes.flat,TITLES.items()):
        for record in records:
            if record['task']!=task:continue
            c=record['curve'];last=c[-1]['rounds'];base=record['base']
            ax.plot([r['rounds']/last for r in c],[100*r['mean_relative_MAE_gain'] for r in c],
                    marker='o',markersize=4,color=COLORS[base],label=LABELS[base],linewidth=1.7)
        ax.axhline(0,color='#777777',linewidth=.7)
        ax.axhline(5,color='#aaaaaa',linewidth=.7,linestyle='--')
        ax.set_title(title,fontsize=11,loc='left');ax.grid(alpha=.15);ax.set_xticks((1/6,.5,1),('1/6','1/2','full'))
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Completed learning curves: development MAE change at a fixed blend weight',fontsize=15)
    fig.supxlabel('Fraction of the full training budget (trees: 3840 rounds; Transformer: 4800 steps)',y=.14,fontsize=10)
    fig.supylabel('Relative MAE reduction versus reference (%)',fontsize=11)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.015,'Mean of fold0/1. One development-selected weight per method, held fixed along the curve.\n'
             '* Rank-task MAE uses a calibrated historical-score reference. Official Spearman is checked separately.\n'
             'Descriptive analysis only; no retuning using confirmation results.',ha='center',fontsize=9)
    dest=ROOT/'docs/v377_v378_learning_curves.png';fig.savefig(dest,dpi=180,bbox_inches='tight');plt.close(fig)
    print(str(dest))


if __name__=='__main__':main()
