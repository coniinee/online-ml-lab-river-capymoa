"""Audit and summarize Step 7–10 measurements without retraining any learner."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .runtime import ROOT

OUT=ROOT/"results/capymoa"
FIG=OUT/"figures"
COLORS=["#276FBF","#E07A26","#269B73","#9B59B6"]
SHORT={"HoeffdingTree":"HT","AdaptiveRandomForest":"ARF","LinearLogistic":"Linear","GaussianNB":"GNB"}


def table(frame,digits=4):
    def val(x):
        if isinstance(x,(list,dict)):return str(x).replace('|','/')
        if pd.isna(x):return '—'
        if isinstance(x,(float,np.floating)):return f'{x:.{digits}f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(val(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def aggregate(frame,groups,columns):
    result=frame.groupby(groups,sort=False)[columns].agg(['mean','std']).reset_index()
    result.columns=['_'.join(filter(None,c)) if isinstance(c,tuple) else c for c in result.columns]
    return result


def save(fig,name):
    for extension in ['png','svg']:fig.savefig(FIG/f'{name}.{extension}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def grouped_bars(frame,value,name,ylabel):
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for ax,dataset in zip(axes.flat,frame.dataset.drop_duplicates()):
        part=frame[frame.dataset==dataset]
        for offset,lib,color in [(-.18,'River',COLORS[0]),(.18,'CapyMOA',COLORS[1])]:
            data=part[part.library==lib].groupby('model',sort=False)[value].agg(['mean','std']).reindex(list(SHORT))
            ax.bar(np.arange(4)+offset,data['mean'],width=.34,yerr=data['std'],capsize=3,label=lib,color=color)
        ax.set_xticks(range(4),list(SHORT.values()));ax.set_title(dataset);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
        if value=='accuracy':ax.set_ylim(0,1)
        else:ax.set_yscale('log')
        ax.legend(fontsize=8)
    fig.suptitle('Controlled cross-library comparison: mean ± seed SD\nLinear = River Adam vs CapyMOA inverse-scaling SGD (closest equivalents)',fontsize=11)
    save(fig,name)


def analyze():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':9})
    config=json.loads((ROOT/'config/capymoa.json').read_text())
    classification=pd.read_csv(OUT/'classification_metrics.csv')
    clustering=pd.read_csv(OUT/'clustering_metrics.csv')
    auto=pd.read_csv(OUT/'automl_metrics.csv')
    from .experiments import signature,compatible
    fingerprint=signature(config)
    records=[json.loads(p.read_text()) for p in (OUT/'runs').glob('*.json')]
    from lab_support.validation import validate_exports
    validate_exports(OUT,records,2)
    expected={'classification':96,'clustering':27,'automl':24}
    checks={stage+'_coverage':len(frame)==expected[stage] for stage,frame in [('classification',classification),('clustering',clustering),('automl',auto)]}
    checks.update({'all_147_records_present':len(records)==147,
                   'all_calculation_signatures_current_or_verified_equivalent':all(compatible(r,config) for r in records),
                   'same_classifier_input_hash_across_libraries':classification.groupby('dataset').data_sha256.nunique().eq(1).all(),
                   'same_automl_input_hash_per_dataset':auto.groupby('dataset').data_sha256.nunique().eq(1).all(),
                   'same_clustering_input_hash_per_dataset':clustering.groupby('dataset').data_sha256.nunique().eq(1).all(),
                   'all_classification_metrics_finite':np.isfinite(classification[['accuracy','macro_f1','kappa','mean_latency_us']]).all().all(),
                   'all_automl_metrics_finite':np.isfinite(auto[['accuracy','macro_f1','kappa','mean_latency_us']]).all().all(),
                   'all_clustering_score_counts_correct':(clustering.n_scored==clustering.n_samples-config['cluster_calibration']).all(),
                   'no_duplicate_run_combinations':all(not d.duplicated(['dataset','model','library','seed']).any() for d in [classification,clustering,auto]),
                   'labels_not_used_to_fit_clusters':all(r['dataset']['labels_used_for_training'] is False for r in records if r['summary']['stage']=='clustering'),
                   'score_bounds':classification.accuracy.between(0,1).all() and auto.accuracy.between(0,1).all()})
    checks={k:bool(v) for k,v in checks.items()}
    checks['sealed_raw_records_and_all_exports_match']=True
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    assert all(checks.values()),checks
    class_cols=['accuracy','macro_f1','kappa','mean_latency_us','p95_latency_us','total_runtime_s','sampled_peak_process_rss_mib']
    class_summary=aggregate(classification,['dataset','model','library'],class_cols)
    class_summary.to_csv(OUT/'classification_summary.csv',index=False)
    cluster_summary=aggregate(clustering,['dataset','model'],['mean_window_ari','mean_window_silhouette','mean_window_coverage','mean_window_mse','mean_latency_us','total_runtime_s','final_micro_clusters'])
    cluster_summary.to_csv(OUT/'clustering_summary.csv',index=False)
    auto_summary=aggregate(auto,['dataset','model'],class_cols)
    auto_summary.to_csv(OUT/'automl_summary.csv',index=False)
    river=classification[classification.library=='River'];capy=classification[classification.library=='CapyMOA']
    pairs=capy.merge(river,on=['dataset','model','seed'],suffixes=('_capy','_river'))
    pairs['accuracy_difference_pp']=100*(pairs.accuracy_capy-pairs.accuracy_river)
    pairs['capy_to_river_latency_ratio']=pairs.mean_latency_us_capy/pairs.mean_latency_us_river
    pairs.to_csv(OUT/'classification_pairs.csv',index=False)
    historical=pd.read_csv(ROOT/'results/river/classification_metrics.csv')
    historical['model']=historical.model.replace({'LogisticAdamOvR':'LinearLogistic'})
    historical=historical.rename(columns={'accuracy':'part1_accuracy','mean_latency_us':'part1_mean_latency_us'})
    pairs.merge(historical[['dataset','model','seed','part1_accuracy','part1_mean_latency_us']],on=['dataset','model','seed']).to_csv(OUT/'part1_historical_comparison.csv',index=False)
    grouped_bars(classification,'accuracy','classification_accuracy','Prequential accuracy')
    grouped_bars(classification,'mean_latency_us','classification_latency','Latency (µs, log scale)')
    curves=pd.read_csv(OUT/'classification_curves.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for ax,ds in zip(axes.flat,config['classification_datasets']):
        for (family,lib),color in zip([('AdaptiveRandomForest','River'),('AdaptiveRandomForest','CapyMOA'),('LinearLogistic','River'),('LinearLogistic','CapyMOA')],COLORS):
            data=curves[(curves.dataset==ds)&(curves.model==family)&(curves.library==lib)].groupby('n_seen').rolling_accuracy.mean()
            ax.plot(data.index,data,label=f'{SHORT[family]} / {lib}',color=color)
        if ds=='SEARecurring':
            for at in [3333,6666]:ax.axvline(at+1,ls='--',color='black',alpha=.6)
        ax.set_title(ds);ax.set_ylim(0,1.02);ax.set_xlabel('Examples processed');ax.set_ylabel('Rolling accuracy (last 500)');ax.xaxis.set_major_locator(MaxNLocator(5,integer=True));ax.legend(fontsize=7);ax.grid(alpha=.15)
    save(fig,'classification_curves')
    cc=pd.read_csv(OUT/'clustering_curves.csv')
    for metric,label,name in [('ari_all','Prequential window ARI','clustering_evolution'),('silhouette_assigned','Silhouette (assigned only)','clustering_silhouette')]:
        fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
        for ax,ds in zip(axes,config['cluster_datasets']):
            for model,color in zip(config['cluster_models'],COLORS):
                d=cc[(cc.dataset==ds)&(cc.model==model)].groupby('n_seen')[metric].mean()
                ax.plot(d.index,d,label=model,color=color)
            if ds=='MovingGaussians':
                for at in [2000,4000]:ax.axvline(at,ls='--',color='black',alpha=.5)
            ax.set_title(ds);ax.set_ylabel(label);ax.set_xlabel('Examples processed');ax.xaxis.set_major_locator(MaxNLocator(4,integer=True));ax.grid(alpha=.2);ax.legend(fontsize=7)
        save(fig,name)
    from .common import clustering_data
    x,y,_=clustering_data('MovingGaussians',config)
    fig,axes=plt.subplots(3,3,figsize=(10.5,9),layout='constrained')
    for row,model in enumerate(config['cluster_models']):
        r=next(r for r in records if r['summary']['stage']=='clustering' and r['summary']['dataset']=='MovingGaussians' and r['summary']['seed']==42 and r['summary']['model']==model)
        for col,at in enumerate([1000,3000,6000]):
            ax=axes[row,col];p=next(p for p in r['snapshots'] if p['n_seen']==at)
            ax.scatter(x[at-300:at,0],x[at-300:at,1],c=y[at-300:at],cmap='viridis',s=5,alpha=.4)
            centers=np.asarray(p['centers'])
            if len(centers):ax.scatter(centers[:,0],centers[:,1],c='black',marker='X',s=90)
            ax.set_title(f'{model}, n={at}');ax.set_xlim(-6,7);ax.set_ylim(-5,7);ax.set_xlabel('x0');ax.set_ylabel('x1')
    fig.suptitle('Synthetic cluster snapshots: black X = common macro centers\nPoint colors show evaluation-only generating components, never training labels',fontsize=11)
    save(fig,'clustering_snapshots')
    ac=pd.read_csv(OUT/'automl_curves.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,ds in zip(axes,config['automl_datasets']):
        for model,color in zip(config['automl_models'],COLORS):
            d=ac[(ac.dataset==ds)&(ac.model==model)].groupby('n_seen').rolling_accuracy.mean()
            ax.plot(d.index,d,label=model,color=color)
        if ds=='SEARecurring':
            for at in [3333,6666]:ax.axvline(at+1,ls='--',color='black',alpha=.5)
        ax.set_title(ds);ax.set_xlabel('Examples processed');ax.set_ylabel('Rolling accuracy (last 500)');ax.set_ylim(0,1.02);ax.xaxis.set_major_locator(MaxNLocator(5,integer=True));ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'automl_curves')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for col,ds in enumerate(config['automl_datasets']):
        for row,(metric,label) in enumerate([('accuracy','Prequential accuracy'),('mean_latency_us','Latency (µs, log scale)')]):
            ax=axes[row,col];d=auto[auto.dataset==ds].groupby('model',sort=False)[metric].agg(['mean','std']).reindex(config['automl_models'])
            ax.bar(range(4),d['mean'],yerr=d['std'],capsize=3,color=COLORS);ax.set_xticks(range(4),d.index,rotation=12,ha='right',fontsize=8);ax.set_title(ds);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
            if row:ax.set_yscale('log')
            else:ax.set_ylim(0,1)
    save(fig,'automl_accuracy_cost')
    final_states=[{**{k:r['summary'][k] for k in ['dataset','model','seed']},'state':json.loads(r['curves'][-1]['selection_state'])} for r in records if r['summary']['stage']=='automl']
    for item in final_states:
        if item['model']=='SuccessiveHalving':
            s=item['state']
            item['derived_model_updates']=s['budget_used']+s['iterations_in_current_rung']*s['active_models']
            assert item['derived_model_updates']==s['actual_model_updates']
            assert s['best_model_index'] in s['active_indices']
            item['budget_counter_scope']='Completed rungs only; derived count additionally includes survivor/current-rung updates.'
    (OUT/'automl_final_states.json').write_text(json.dumps(final_states,indent=2)+'\n')
    build_findings(config,classification,clustering,auto,pairs,checks)
    print(f'Analysis complete: {len(checks)} checks passed; 8 figures; English findings generated.')


def build_findings(config,classification,clustering,auto,pairs,checks):
    cm=classification.groupby(['dataset','model','library'],sort=False)[['accuracy','macro_f1','kappa','mean_latency_us','total_runtime_s']].mean().reset_index()
    km=clustering.groupby(['dataset','model'],sort=False)[['mean_window_ari','mean_window_silhouette','mean_window_coverage','mean_latency_us']].mean().reset_index()
    am=auto.groupby(['dataset','model'],sort=False)[['accuracy','macro_f1','mean_latency_us','total_runtime_s']].mean().reset_index()
    arf_reference=cm[(cm.library=='CapyMOA')&(cm.model=='AdaptiveRandomForest')&(cm.dataset.isin(config['automl_datasets']))][['dataset','model','accuracy','macro_f1','mean_latency_us','total_runtime_s']].copy()
    arf_reference['model']='ARF (Step 8 reference)'
    auto_with_reference=pd.concat([am,arf_reference],ignore_index=True)
    auto_with_reference.to_csv(OUT/'automl_with_arf_reference.csv',index=False)
    effects=pairs.groupby(['dataset','model'],sort=False)[['accuracy_difference_pp','capy_to_river_latency_ratio']].mean().reset_index()
    from scripts.english_findings import write_capy
    write_capy(ROOT)
