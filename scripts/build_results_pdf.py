"""Create one results PDF only after the main studies and real C job complete.

Uses saved measurements; does not run analysis, inference, or change results.
Outputs are immutable: use a new --out directory for a later report revision.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.make_recovery_proposal import styles,inline as base_inline,WIDTH,MUTED,RULE
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,Preformatted,KeepTogether


def read(path):return json.loads(Path(path).read_text())


def fmt(value,digits=2):
    return 'unavailable' if value is None else f'{value:.{digits}f}'


def mean(values):
    return sum(values)/len(values) if values and all(v is not None for v in values) else None


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n\n'


def interval(value,digits=2):
    return f"{fmt(value.get('point'),digits)} [{fmt(value.get('lo'),digits)}, {fmt(value.get('hi'),digits)}]"


def auc_details(cells,kind):
    rows=[]
    for c in cells:
        scores=c['features'] if kind=='M' else c['scores']
        key=f"{c['arm']} / {c['checkpoint']} / {c['offset']}"+(f" / {c['target']}" if kind!='M' else '')
        for feature,value in scores.items():
            if value.get('point') is None:continue
            rows.append([key+' / '+feature,fmt(value['point'],3),
                         f"[{fmt(value['lo'],3)}, {fmt(value['hi'],3)}]",
                         f"{value['n_local']} / {value['n_slip']}",value['valid_bootstraps']])
        delta_key='verbal_minus_logit' if kind=='M' else 'lp_minus_verbal'
        delta=c.get(delta_key,{})
        if delta.get('point') is not None:
            support=scores['verbal']
            rows.append([key+' / '+delta_key,fmt(delta['point'],3),
                         f"[{fmt(delta['lo'],3)}, {fmt(delta['hi'],3)}]",
                         f"{support['n_local']} / {support['n_slip']}",delta['valid_bootstraps']])
    if not rows:return 'No identifiable strict cells.\n\n'
    return ('Feature rows show AUROC. Named subtraction rows show paired AUROC differences in the direction stated.\n\n'+
            table(['Cell / score','Value','95% CI','n positive / negative','Valid draws'],rows))


def a0_section(data):
    text='## Exploratory A0 metacognition — existing data only\n\n'
    text+=f"This exploratory analysis uses {data['n_trajectories']} trajectories / {data['n_rows']} task rows and {data['n_boot']} trajectory-bootstrap draws. It is not part of fresh confirmation. Position, errors in context and performance are confounded; pooled AUROC is not a causal monitoring test.\n\n"
    rows=[]
    for cell in data['bins']:
        for key in ['verbal','ptrue','fore_none','fore_true','fore_placebo','higher','same','lower']:
            t=cell[key];scale=100 if key in ['ptrue','higher','same','lower'] else 1
            ci='unavailable' if t['mean'] is None else f"{fmt(scale*t['mean'])} [{fmt(scale*t['lo'])}, {fmt(scale*t['hi'])}]"
            actual=cell['matched_realized'].get(key)
            rows.append([cell['bin']+' / '+key,ci,t['n'],fmt(100*actual['mean']) if actual and actual['mean'] is not None else 'unavailable'])
    text+=table(['Position / signal','Mean [95% CI], 0–100','n','Matched next-10 accuracy %'],rows)
    text+=table(['Checkpoint','Verbal / 100','No-evidence forecast %','Truth / placebo forecast %','Actual next-10 %'],
                [[v['turn'],fmt(v['verbal']['mean']),fmt(v['fore_none']['mean']),f"{fmt(v['fore_true']['mean'])} / {fmt(v['fore_placebo']['mean'])}",fmt(100*v['realized']['mean'])] for v in data['checkpoints']])
    text+='A0 no-evidence forecasts are high early, then reach zero from checkpoint 100; “about 100” is not an appropriate description of the full trajectory. Awareness category probabilities and P(Yes) are expressed as percentages in this section.\n\n'
    rows=[]
    for label,scores in data['auc'].items():
        for key,t in scores.items():rows.append([label+' / '+key,interval(t,3),t['n_local'],t['n_slip']])
    text+=table(['Position / score','Local-correctness AUROC [95% CI]','n locally correct','n numeric slips'],rows)
    matched=data['matched']
    text+=f"Exact-position clean-versus-previous-error comparison: error−clean verbal difference {fmt(matched['error_minus_clean'])}, 95% CI [{fmt(matched['lo'])}, {fmt(matched['hi'])}]. Positions with both groups receive equal weight; sparse cells have n<30. No-current-error history definition: {matched['definition']}.\n\n"
    text+=table(['Turn / prior history','n','Verbal mean','95% CI','Sparse n<30'],[[f"{v['turn']} / {v['history']}",v['n'],fmt(v['mean']),f"[{fmt(v['lo'])}, {fmt(v['hi'])}]",str(v['sparse'])] for v in matched['cells']])
    sensitivity=data['matched_inclusive']
    text+=f"Inclusive-current-error sensitivity: difference {fmt(sensitivity['error_minus_clean'])}, 95% CI [{fmt(sensitivity['lo'])}, {fmt(sensitivity['hi'])}], valid bootstrap draws {sensitivity['valid_bootstraps']}. Primary prior-error comparison valid draws: {matched['valid_bootstraps']}. These are sparse descriptive associations, not causal effects.\n\n"
    for name,caption in [('meta_position.png','Exploratory A0 position curves'),('meta_matched.png','A0 position-matched history comparison'),('meta_calibration.png','A0 forecast calibration by checkpoint')]:
        text+=f'![{caption}](figures/A0/{name})\n\n'
    return text


def inline(text):
    text=re.sub(r'\[([^\]]+)\]\((?!https?://)[^)]+\)',r'\1',text)
    return re.sub(r'`([^`]+)`',r'<font name="DVM" size="7.7">\1</font>',base_inline(text))


def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(RULE)
    canvas.line(18*mm,15*mm,A4[0]-18*mm,15*mm)
    canvas.setFillColor(MUTED);canvas.setFont('DV',7)
    canvas.drawString(18*mm,10*mm,'Catching Silent Collapse | Experimental results | Private review')
    canvas.drawRightString(A4[0]-18*mm,10*mm,str(doc.page));canvas.restoreState()


def render(source,output):
    st=styles();pdfmetrics.registerFont(TTFont('DVM','/usr/share/fonts/dejavu/DejaVuSansMono.ttf'))
    st['section']=ParagraphStyle('section',parent=st['heading'],fontSize=12,leading=16)
    st['code']=ParagraphStyle('code',parent=st['body'],fontName='DVM',fontSize=6.5,leading=9)
    flow=[];lines=source.read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line.startswith('```'):
            block=[]
            while i<len(lines) and not lines[i].strip().startswith('```'):
                block.append(lines[i]);i+=1
            i+=1;flow.extend([Preformatted('\n'.join(block),st['code']),Spacer(1,6)]);continue
        if line.startswith('|'):
            block=[line]
            while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i].strip());i+=1
            rows=[[c.strip() for c in l.strip('|').split('|')] for l in block]
            rows=[r for r in rows if not all(re.fullmatch(r':?-+:?',c) for c in r)]
            n=len(rows[0]);weights={3:[.35,.3,.35],4:[.18,.28,.27,.27],5:[.30,.13,.25,.17,.15]}.get(n,[1/n]*n)
            data=[[Paragraph(inline('**'+c+'**' if j==0 else c),st['cell']) for c in row] for j,row in enumerate(rows)]
            t=Table(data,colWidths=[WIDTH*w/sum(weights) for w in weights],repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4ebf3')),
                                  ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),
                                  ('GRID',(0,0),(-1,-1),.3,RULE),('LEFTPADDING',(0,0),(-1,-1),5),
                                  ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            flow.extend([t,Spacer(1,8)]);continue
        picture=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if picture:
            path=source.parent/picture[2];im=Image(str(path));scale=min(WIDTH/im.imageWidth,300/im.imageHeight)
            im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
            flow.append(KeepTogether([Paragraph(inline(picture[1]),st['heading']),im,Spacer(1,8)]));continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));style=st['title'] if level==1 else st['section'] if level==2 else st['heading']
            flow.append(Paragraph(inline(line[level:].strip()),style));continue
        paragraph=[line]
        while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','![','```','- ')):
            paragraph.append(lines[i].strip());i+=1
        if line.startswith('- '):flow.append(Paragraph(inline(' '.join(paragraph)[2:]),st['bullet'],bulletText='•'))
        else:flow.append(Paragraph(inline(' '.join(paragraph)),st['body']))
    doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,
                          topMargin=17*mm,bottomMargin=20*mm,title='Catching Silent Collapse — All Experimental Results',
                          author='Catching Silent Collapse project',subject='Saved experimental measurements and registered decisions')
    doc.build(flow,onFirstPage=footer,onLaterPages=footer)


def m_section(name,data,primary,image_base):
    text=f'## {name}\n\nReported decision: **{primary["verdict"]}**.\n\n'
    tests=primary.get('tests',{});contrasts=primary.get('contrasts',{})
    text+=table(['Contrast','Estimate, points','95% CI'],[[key,fmt(t['diff']),f"[{fmt(t['lo'])}, {fmt(t['hi'])}]"] for key,t in contrasts.items()])
    text+=table(['HF test','Raw bootstrap p','Holm p','Interval rule','Supported'],
                [[k,fmt(t['p'],6),fmt(t['p_holm'],6),str(t['ci_rule']),str(t['supported'])] for k,t in tests.items()])
    text+=f"Mean confidence is nonincreasing across intermediate doses at each checkpoint: {data.get('dose_monotone',{})}. All intermediate cells at the accuracy floor: {data.get('intermediates_all_at_floor')}.\n\n"
    text+='Stored HF p=0 means below empirical resampling resolution, not mathematical zero. Superiority also requires the directional interval rule. Dose equivalence uses its 90% interval and the ±10-point band.\n\n'
    text+=table(['Equivalence test','90% interval','Registered bounds'],[[k,f"[{fmt(t['lo90'])}, {fmt(t['hi90'])}]",'(-10, +10)'] for k,t in tests.items() if 'lo90' in t])
    arms=[]
    for arm in ['L0','L25','L50','L75','L100','S0','S100']:
        cells=[x for x in data['means'] if x['arm']==arm]
        arms.append([arm,fmt(mean([x['verbal'] for x in cells])),fmt(mean([x['fore_none'] for x in cells])),fmt(100*mean([x['realized'] for x in cells]))])
    text+=table(['Arm','Verbal / 100','Forecast %','Actual accuracy %'],arms)
    text+=table(['Checkpoint','Arm','n','Confidence / forecast','Accuracy %'],
                [[x['checkpoint'],x['arm'],x['n'],f"{fmt(x['verbal'])} / {fmt(x['fore_none'])}",fmt(100*x['realized'])] for x in data['means']])
    text+=table(['Checkpoint / arm','Awareness optimism %','Invalid verbal','Invalid forecast','Invalid awareness'],
                [[f"{x['checkpoint']} / {x['arm']}",fmt(100*x['aware']) if x['aware'] is not None else 'unavailable',x['invalid']['verbal'],x['invalid']['fore_none'],x['invalid']['aware']] for x in data['means']])
    text+=table(['Signal','Calibration slope [CI]','Intercept [CI]'],
                [[k,f"{fmt(v.get('slope'),3)} [{fmt(v.get('lo'),3)}, {fmt(v.get('hi'),3)}]",
                  f"{fmt(v.get('intercept'),3)} [{fmt(v.get('intercept_lo'),3)}, {fmt(v.get('intercept_hi'),3)}]"] for k,v in data['calibration'].items()])
    text+='Calibration regressions condition on checkpoint and bootstrap whole trajectories. Verbal confidence concerns the current answer; awareness is ordinal. Only the forecast is a next-ten probability estimate. History truncation also removes evidence; it is not a pure token-count intervention.\n\n'
    invalid={k:sum(v['invalid'][k] for v in data['means']) for k in ['verbal','fore_none','aware']}
    text+=f'Invalid checkpoint probes: {invalid}. Floor cells (accuracy ≤5%): '+', '.join(f"{v['arm']}@{v['checkpoint']}" for v in data['means'] if v['floor'])+'.\n\n'
    if 'confidence' in image_base:text+=f"![Confidence, forecast and realized performance]({image_base['confidence']})\n\n"
    if 'curves' in image_base:text+=f"![Fresh-sequence confirmation M curves]({image_base['curves']})\n\n"
    for key,caption in [('forecast','Forecast and realized accuracy'),('awareness','Awareness optimism and realized accuracy'),('dose_response','Confidence dose-response; endpoint descriptive')]:
        if key in image_base:text+=f'![{caption}]({image_base[key]})\n\n'
    auc=data.get('fixed_position_auc',[])
    count=sum(v['features'].get('lp_min',{}).get('point') is not None for v in auc)
    text+=f'Strict minimum-logprob AUROC support: {count}/{len(auc)} checkpoint/offset cells. Undefined cells are not zeros and are not silently pooled. Full feature intervals are saved in the analysis JSON.\n\n'
    text+='### Identifiable strict localization cells\n\nVerbal and logit comparisons use their paired common observations. P(Yes) support may differ; inspect the saved per-feature class counts before comparison.\n\n'
    text+=auc_details(auc,'M')
    return text


def mr_section(name,data,primary,image_base):
    text=f'## {name}\n\nRegistered verdict: **{primary["verdict"]}**.\n\n'
    text+=table(['HG test','Estimate, points','95% CI','Holm p','Supported'],
                [[k,fmt(v['mean']),f"[{fmt(v['lo'])}, {fmt(v['hi'])}]",fmt(v['p_holm'],6),str(v['supported'])] for k,v in primary.get('tests',primary.get('diagnostic_estimates',{})).items()])
    tests=primary.get('tests',primary.get('diagnostic_estimates',{}))
    if 'HG2' in tests:
        eq=tests['HG2'];text+=f"HG2 90% equivalence interval: [{fmt(eq['lo90'])}, {fmt(eq['hi90'])}]; bounds (-5, +5).\n\n"
        if 'diagnostic_estimates' in primary:text+='Calibration diagnostics do not issue a confirmatory verdict.\n\n'
    text+='HG1: repaired−wrong-state accuracy >0. HG2: repaired−clean accuracy equivalent within ±5. HG3: clean−repaired forecast >10. HG4: repaired accuracy−forecast >10. All four must pass. A lower forecast than an overconfident reference is insufficient evidence of underconfidence.\n\n'
    rows=data['checkpoint_summary'];arms=[]
    for arm in ['MR_A','MR_B','MR_C','MR_D']:
        cells=[r for r in rows if r['arm']==arm]
        arms.append([arm,fmt(mean([x['accuracy'] for x in cells])),fmt(mean([x['pre_forecast'] for x in cells])),fmt(mean([x['forecast'] for x in cells]))])
    text+=table(['Arm','Accuracy %','Pre-cue forecast %','Post-cue forecast %'],arms)
    text+=table(['Arm / checkpoint','Accuracy %','Forecast %','MAE','Missing'],
                [[f"{x['arm']} / {x['checkpoint']}",fmt(x['accuracy']),fmt(x['forecast']),fmt(x['mae']),x['missing_forecasts']] for x in rows])
    text+=f"Primary overall floor flag: {primary.get('floor','see JSON')}. Missing primary forecasts: {sum(x['missing_forecasts'] for x in rows)}. Out-of-range primary forecasts: {sum(x['out_of_range'] for x in rows)}.\n\n"
    text+=table(['Arm','Calibration slope [CI]','Intercept [CI]'],
                [[k,f"{fmt(v.get('slope'),3)} [{fmt(v.get('lo'),3)}, {fmt(v.get('hi'),3)}]",
                  f"{fmt(v.get('intercept'),3)} [{fmt(v.get('intercept_lo'),3)}, {fmt(v.get('intercept_hi'),3)}]"] for k,v in data['calibration'].items()])
    text+=table(['Arm / checkpoint','Awareness higher %','Same %','Lower %','Valid n'],
                [[f"{x['arm']} / {x['checkpoint']}"]+[fmt(100*p) for p in x['awareness']]+[x['awareness_n']] if x['awareness'] is not None else [f"{x['arm']} / {x['checkpoint']}",'unavailable','unavailable','unavailable',0] for x in rows])
    text+=table(['Interaction','Estimate, points','95% CI'],[[k,fmt(v['mean']),f"[{fmt(v['lo'])}, {fmt(v['hi'])}]"] for k,v in data['interactions'].items()])
    for target in ['global','local']:
        cells=[c for c in data['auroc'] if c['target']==target];n=sum(c['scores']['lp_min']['point'] is not None for c in cells)
        text+=f'{target.capitalize()} minimum-logprob AUROC support: {n}/{len(cells)} strict cells. '
    text+='Calibration, interactions and discrimination are secondary and do not change the registered primary verdict.\n\n'
    text+='### Identifiable strict discrimination cells\n\n'
    text+=auc_details(data['auroc'],'MR')
    for label,file in [('Accuracy by checkpoint','accuracy.png'),('Forecast versus realized accuracy','forecasts.png'),('Registered HG effects and intervals','hg_effects.png'),('Calibration and discrimination','secondary.png')]:
        text+=f'![{label}]({image_base}/{file})\n\n'
    return text


def build_report(confirmation_root,out):
    main=read(ROOT/'results/main_studies_completion_20261118.json');state=read(confirmation_root/'status.json')
    if main['status']!='complete' or state['status']!='complete' or state['signature']['mock']:
        raise ValueError('All real main and confirmation experiments must finish before PDF creation')
    c=read(confirmation_root/'analysis/analysis.json')
    if not c['eligibility']['eligible']:raise ValueError('Confirmation is not eligible; inspect its saved report first')
    m=read(ROOT/'results/meta_HF_main_20261118/meta_experiment.json');mr=read(ROOT/'results/recovery_HG_gpu0_main_20261118/analysis.json')
    images=out/'figures';images.mkdir()
    import shutil
    shutil.copyfile(ROOT/'results/meta_HF_main_20261118/figures/confidence.png',images/'main_M_confidence.png')
    for name in ['forecast','awareness','dose_response']:
        shutil.copyfile(ROOT/f'results/meta_HF_main_20261118/figures/{name}.png',images/f'main_M_{name}.png')
    (images/'A0').mkdir()
    for name in ['meta_position.png','meta_matched.png','meta_calibration.png']:
        shutil.copyfile(ROOT/'results/figures'/name,images/'A0'/name)
    # Reuse the plotter with C's explicit eligibility label; retain every saved number.
    from scripts.analyze_confirmation import plot_m
    plot_m(dict(c['M_base_analysis'],status=c['status']+'; C-HF: '+c['C_HF']['verdict']),images)
    (images/'M_curves.png').rename(images/'C_M_curves.png')
    for label,base in [('main_MR',ROOT/'results/recovery_HG_gpu0_main_20261118'),('C_MR',confirmation_root/'analysis')]:
        d=images/label;d.mkdir()
        for name in ['accuracy.png','forecasts.png','hg_effects.png','secondary.png']:shutil.copyfile(base/name,d/name)
    text='# Catching Silent Collapse — All Experimental Results\n\n'
    text+='Results report generated from saved data after all approved experiments. This is not a paper or submission.\n\n'
    text+=f"Main study completion: {main['completed_at']}. Confirmation completion: {state['updated']}.\n\n"
    text+=table(['Experiment','Trajectories','Conditions','Task answers'],[['Main M',32,1568,15680],['Main MR',32,896,8960],['Fresh confirmation M',32,1568,15680],['Fresh confirmation MR',32,896,8960],['Core experiment total; excludes calibration/A0','64 unique sequences',4928,49280]])
    text+='Main ids 20–51 were previously examined in A0. Independent confirmation ids 12000–12031 use fresh seeds 2000–2031 and the same dictionary. All checkpoints are 10,25,50,100,150,200,250, with ten future task answers each. Calibration is excluded from confirmatory estimates. Side probes are excluded from task-answer counts.\n\n'
    config=read(confirmation_root/'M/manifest.json')['configs'][0]
    resources=read(ROOT/'results/recovery_resources_complete_20261117.json')
    text+='### Protocol and score definitions\n\n'
    text+=table(['Setting','Saved value'],[
        ['Model',config['model_id']],['Model revision',config['model_revision']],
        ['Published dataset revision',resources['dataset_revision']],
        ['Task','K=1 key/turn; T=300; same dictionary in all cohorts'],
        ['Inference','bf16; greedy constrained integer answers; no thinking; independent probe forks'],
        ['Runtime',f"context {config['max_model_len']}; memory fraction {config['util']}; one active sequence; 512-token chunked prefill; eager"],
        ['Original scientific registration','1d11bc5; local commit before main inference'],
        ['Original MR execution amendment','e45f9df; before MR main inference'],
        ['Fresh confirmation registration',state['signature']['registration_commit']]])
    text+='M arms L0/L25/L50/L75/L100 supply long histories with the named percentage of wrong totals; S0/S100 show the last five turns of their long counterparts. Every wrong supplied total is truth+1 from the approved generator. P=S0−L0 verbal confidence; length L=−P. D is minus the confidence OLS slope over doses 0,.25,.50,.75, expressed per full dose range. The L0−L100 endpoint is descriptive at the accuracy floor.\n\n'
    text+='MR_A=clean history/true supplied state; MR_B=error history/wrong state; MR_C=the same error history/true state; MR_D=the same clean history/wrong state. One neutral state cue precedes continuation. The first local label uses that supplied state; global truth does not change. Pre-cue and post-cue forecasts and actual task answers are independent forks.\n\n'
    text+='C was registered after the original main results were known, retaining every original condition and decision rule. Fresh-sequence allocation used fixed seeds and content hashes rather than model outcomes. These are independent synthetic sequences within recorded project use, not an independent model or a guarantee about unknown training exposure.\n\n'
    text+=table(['Decision','Original main','Fresh confirmation'],[['HF',m['HF']['verdict'],c['C_HF']['verdict']],['HG',mr['primary']['verdict'],c['C_HG']['verdict']]])
    text+='Verbal confidence evaluates the current answer. Forecasts predict the percentage of globally correct next-ten answers. Awareness is ordinal. For local-slip AUROC, ok+inherited are positive and numeric slips negative, excluding nonanswers. MR also reports global-correctness AUROC. Secondary intervals are descriptive and do not create additional primary decisions.\n\n'
    text+='Each study uses 10,000 whole-trajectory bootstrap draws and the registered separate four-test Holm families. Main and confirmation estimates are reported separately; no unregistered pooled decision is made.\n\n'
    text+='Scores: verbal confidence rates the current answer; forecasts predict the percentage of globally correct next-ten answers; awareness compares performance with the start. Local correctness includes inherited wrong totals that are arithmetically consistent with the preceding state; numeric slips are fresh local errors. MR also reports global correctness, with nonanswers counted wrong. Strict AUROC cells require at least 15 per class. Secondary intervals are descriptive, not additional adjusted primary claims.\n\n'
    text+=m_section('Original main M',m,m['HF'],{key:f'figures/main_M_{key}.png' for key in ['confidence','forecast','awareness','dose_response']})
    text+=mr_section('Original main MR',mr,mr['primary'],'figures/main_MR')
    text+=m_section('Fresh confirmation C-M',c['M_base_analysis'],c['C_HF'],{'curves':'figures/C_M_curves.png'})
    text+=mr_section('Fresh confirmation C-MR',c['MR_base_analysis'],c['C_HG'],'figures/C_MR')
    text+='## Calibration and technical checks — exploratory only\n\n'
    text+=table(['Calibration run','Trajectories','Conditions','Task answers'],[['MR initial smoke',2,8,80],['MR full feasibility',8,224,2240],['M full calibration',8,392,3920]])
    text+='These calibration observations informed feasibility/runtime and prospective precision review. They are not confirmation observations. Precision simulations resample small-pilot residuals; doubling their scale means fourfold variance, not a guarantee of power. Mock values are software fixtures and are excluded from all experimental tables.\n\n'
    pilot=read(ROOT/'results/recovery_gpu1_pilot_20261117/analysis.json')
    text+=mr_section('Exploratory MR calibration pilot',pilot,pilot['primary'],'figures/calibration_MR')
    d=images/'calibration_MR';d.mkdir()
    for name in ['accuracy.png','forecasts.png','hg_effects.png','secondary.png']:
        shutil.copyfile(ROOT/'results/recovery_gpu1_pilot_20261117'/name,d/name)
    pilot_m=read(ROOT/'results/meta_gpu1_calibration_20261118/meta_experiment.json')
    text+=m_section('Exploratory M calibration',pilot_m,pilot_m['HF'],{})
    text+=a0_section(read(ROOT/'results/meta_analysis.json'))
    text+='## Interpretation and reproducibility limits\n\n'
    text+='The long/short comparison changes available evidence as well as tokens. Equal turn counts do not ensure exact token-count matching; wrong totals can change number widths and correction transitions. All injected errors are controlled +1 totals in teacher-forced histories. The task and model are narrow; no internal metacognitive mechanism or cross-model generality is established. Relative repair to a low clean baseline is not near-perfect execution. Sparse AUROC cells restrict localization claims. Missing or invalid estimates are reported as unavailable.\n\n'
    text+='Original M used GPU 1; original MR used GPU 0 after documented hardware/scheduling amendments. GPU 2 failed initialization before MR task generation because service PATH omitted ninja. The corrected PATH uses the already installed tool. Confirmation uses GPU 0 with its own frozen sample/source/config manifest. Model weights and original arm outputs were preserved.\n\n'
    text+='All raw archives, checkpoint envelopes, JSONL rows, trajectories, analysis JSON, class-support counts, source snapshots and failure logs remain saved. Statistical decisions and missingness must be read alongside the arm curves. Original registration, execution amendments and C registration are separate records. Nothing is submitted or sent externally by this report.\n\n'
    text+='## Saved measurement sources\n\n'
    text+=table(['Study','Raw / checkpoint root','Analysis'],[['M','runs/meta_HF_main_20261118/','results/meta_HF_main_20261118/'],['MR','runs/recovery_HG_gpu0_main_20261118/','results/recovery_HG_gpu0_main_20261118/'],['C',str(confirmation_root.relative_to(ROOT)),str((confirmation_root/'analysis').relative_to(ROOT))]])
    source=out/'all_experimental_results.md';source.write_text(text)
    render(source,out/'all_experimental_results.pdf')
    input_paths=[ROOT/'results/main_studies_completion_20261118.json',ROOT/'results/meta_HF_main_20261118/meta_experiment.json',ROOT/'results/recovery_HG_gpu0_main_20261118/analysis.json',ROOT/'results/meta_analysis.json',ROOT/'results/recovery_gpu1_pilot_20261117/analysis.json',ROOT/'results/meta_gpu1_calibration_20261118/meta_experiment.json',confirmation_root/'status.json',confirmation_root/'analysis/analysis.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    hashes['scripts/build_results_pdf.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'report_inputs.json').write_text(json.dumps(hashes,indent=2)+'\n')


def main():
    os.environ['TZ']='UTC'
    if hasattr(time,'tzset'):time.tzset()
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--confirmation-root',type=Path,default=ROOT/'runs/confirmation_C_main_20261118')
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    # Admission checks precede any report output, including an output directory.
    state=read(a.confirmation_root/'status.json')
    if state['status']!='complete' or state['signature']['mock']:ap.error('Real confirmation is not complete')
    if a.out.exists():ap.error('Output exists; choose a new report directory')
    a.out.mkdir(parents=True);build_report(a.confirmation_root.resolve(),a.out.resolve())
    print(a.out/'all_experimental_results.pdf')


if __name__=='__main__':main()
