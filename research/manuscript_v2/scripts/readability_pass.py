#!/usr/bin/env python3
"""Propose authorized SVG readability edits as JSON, without writing files.

Word wrapping preserves frozen wording. A1 uses two processing rows solely to
make the existing causal path readable; no module or connection is removed.
"""
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
FIG=ROOT/'论文投稿/manuscript_v2/figures'
NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)


def find(root,id):
    return next(e for e in root.iter() if e.get('id')==id)


def wrapped(root,id,lines,ys,size=None,box=None):
    element=find(root,id)
    for child in list(element): element.remove(child)
    element.text=None
    if size:element.set('font-size',str(size))
    if box:element.set('data-box',box)
    element.set('y',str(ys[0]))
    for line,y in zip(lines,ys):
        child=ET.SubElement(element,f'{{{NS}}}tspan',{'x':element.get('x'),'y':str(y)})
        child.text=line


def propose(name):
    source=subprocess.check_output(['git','show',f'56ab6cee930da08637c3fa7c6b36d5cb22ccf09f:论文投稿/manuscript_v2/figures/{name}.svg'],cwd=ROOT,text=True)
    root=ET.fromstring(source)
    if name=='fig1_paired_protocol':
        root.set('viewBox','0 0 1752 941');root.set('height',f'{6.8*941/1752:.9f}in')
        find(root,'white-background').set('width','1752')
        for e in root.iter(f'{{{NS}}}text'):
            if float(e.get('font-size'))<26:e.set('font-size','26')
            for span in e.iter(f'{{{NS}}}tspan'):
                if span.get('font-size')=='70%':span.set('font-size','22')
        for n in range(1,5):find(root,f'evidence-{n}').set('width','450')
        for e in root.iter(f'{{{NS}}}text'):
            if 'data-box' in e.attrib and float(e.get('x','0'))>1340:
                x,y,w,h=map(float,e.get('data-box').split(','));e.set('data-box',f'{x},{y},376,{h}')
        wrapped(root,'label-29',['Weaker-teacher','counterexample'],[541,571],26,'1349,519,376,129')
        find(root,'label-30').set('y','601');find(root,'label-31').set('y','628');find(root,'label-32').set('y','655')
        find(root,'evidence-3').set('height','157')
        for n in [30,31,32]:find(root,f'label-{n}').set('data-box','1349,519,376,147')
        find(root,'label-15').set('data-box','922,397,313,58')
        for e in root.iter(f'{{{NS}}}rect'):
            if e.get('x')=='945' and e.get('y')=='470':
                e.set('x','932');e.set('width','295')
        find(root,'label-16').set('data-box','934,475,291,42')
        # Preserve the approved objective label here; manuscript expands its
        # distillation term unambiguously in Equation (1).
    elif name=='fig3_teacher_quality':
        for e in root.iter(f'{{{NS}}}text'):
            if float(e.get('font-size'))<25.5:e.set('font-size','25.5')
        wrapped(root,'label-15',['(alternative teacher','state)'],[602,630],25.5,'350,579,250,61')
        find(root,'teacher-a').set('height','170')
        for e in root.iter(f'{{{NS}}}rect'):
            if e.get('x')=='435' and e.get('fill') in ['#f6a4ad','#ec828b','#c55d65']:
                e.set('y',str(float(e.get('y'))+28))
        find(root,'weak-quality').set('x','1060');find(root,'weak-quality').set('width','355')
        for n in [21,22]:find(root,f'label-{n}').set('data-box','1065,277,345,62')
        find(root,'students').set('x','29');find(root,'students').set('width','253')
        find(root,'label-8').set('data-box','31.5,429,248,75')
        find(root,'label-9').set('data-box','31.5,429,248,75')
        for e in root.iter(f'{{{NS}}}path'):
            if e.get('d','').startswith('M 275 510'):e.set('d',e.get('d').replace('M 275 510','M 283 510',1))
        find(root,'label-27').set('data-box','1442.5,531,295,41')
    elif name=='fig4_ensemble_control':
        for e in root.iter(f'{{{NS}}}text'):
            if float(e.get('font-size'))<29:e.set('font-size','29')
        for id in ['label-39','label-40','label-43']:root.remove(find(root,id))
        for id,lines in [('label-19',['Heterogeneous TG','ensemble']),('label-24',['Homogeneous TT','ensemble'])]:
            wrapped(root,id,lines,[202,235],29,'1164,178,374,64' if id=='label-19' else '1570,178,369,64')
        for n in [20,25]:
            find(root,f'label-{n}').set('y','267');find(root,f'label-{n}').set('data-box','1164,242,374,34' if n==20 else '1570,242,369,34')
        for e in root:
            if e.tag==f'{{{NS}}}rect' and e.get('x') in ['1162','1568']:e.set('height','212')
            if e.tag==f'{{{NS}}}rect' and e.get('x') in ['1246','1652']:
                e.set('y',str(float(e.get('y'))+22))
            if e.tag==f'{{{NS}}}path' and e.get('d','').startswith(('M 1324','M 1368','M 1730','M 1774')):
                e.set('transform','translate(0 22)')
        for n in [21,22,23,26,27,28]:
            e=find(root,f'label-{n}');e.set('y',str(float(e.get('y'))+22))
            x,y,w,h=map(float,e.get('data-box').split(','));e.set('data-box',f'{x},{y+22},{w},{h}')
        for e in root.iter(f'{{{NS}}}path'):
            if e.get('d','').startswith('M 1352 364'):e.set('d','M 1352 386 L 1352 399')
            if e.get('d','').startswith('M 1752 365'):e.set('d','M 1752 386 L 1752 399')
        for e in root.iter(f'{{{NS}}}rect'):
            if e.get('x')=='1152' and e.get('y')=='596':e.set('height','83')
        wrapped(root,'label-42',['TT yields lower student NLL than TG','for all three matched students.'],[629,663],29,'1162,602,774,72')
    return ET.tostring(root,encoding='unicode')+'\n'


def architecture():
    # Semantic reconstruction, not raster tracing. Shared font/line palette.
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="6.8in" height="2.798647in" viewBox="0 0 1774 730" role="img" aria-labelledby="figure-title figure-description">',
      '<title id="figure-title">Teacher architecture</title>',
      '<desc id="figure-description">Corrected causal Grassmann processing. Local direction z_(t−Δ) to z_t; normalized Plücker coordinates, geometric projection, valid-window mean, feature-dependent gated mixing and LM head. Generic late-logit fusion retains free alpha.</desc>',
      '<style>text{font-family:"Liberation Sans",Arial,Helvetica,sans-serif;fill:#0b1c35}</style>',
      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9Z" fill="#152a43"/></marker></defs>',
      '<rect id="white-background" width="1774" height="730" fill="white"/>']
    label=0
    def box(id,x,y,w,h,fill,stroke):
        parts.append(f'<rect id="{id}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    def text(x,y,words,size=28,bold=False,owner=None):
        nonlocal label
        label+=1
        own=f' data-box="{owner}"' if owner else ''
        parts.append(f'<text id="label-{label}" x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="middle"{own}>{words}</text>')
    def arrow(d):parts.append(f'<path d="{d}" fill="none" stroke="#152a43" stroke-width="2.2" stroke-linejoin="round" marker-end="url(#arrow)"/>')
    def math(s):return s
    def sub(s):return f'<tspan baseline-shift="sub" font-size="22">{s}</tspan>'
    def sup(s):return f'<tspan baseline-shift="super" font-size="22">{s}</tspan>'
    box('input',23,60,175,630,'#f5f6f7','#919aa5')
    text(110,103,'Input tokens',28,True)
    text(110,143,'x'+sub('1…L'),32)
    for y,s in [(200,'1'),(300,'2'),(400,'3'),(600,'L')]:
        box('token-'+s,65,y,90,58,'#f3f4f6','#aeb5c0');text(110,y+39,'x'+sub(s),32)
    text(110,537,'⋮',32)
    box('transformer-branch',270,55,950,182,'#f0f7ff','#368bea')
    text(560,96,'Transformer branch (T)',32,True)
    for id,x,w,lines,fill in [('token-embedding-t',295,150,['Token','Embedding'],'#ffe8e9'),('transformer-blocks',480,260,['Transformer Blocks','(× N'+sub('T')+')'],'#e1effe'),('layer-norm',775,185,['Layer Norm'],'#deefff'),('lm-head-t',995,200,['LM Head'],'#eae3fa')]:
        box(id,x,125,w,88,fill,'#368bea' if id!='lm-head-t' else '#a188e8')
        for i,s in enumerate(lines):text(x+w/2,160+33*i if len(lines)>1 else 177,s,32 if 'tspan' in s else 28,owner=f'{x+3},128,{w-6},82')
    for a,b in [(445,480),(740,775),(960,995),(1195,1240)]:arrow(f'M{a+1} 169H{b-2}')
    box('t-logits',1240,123,130,92,'#f0f7ff','#368bea');text(1305,158,'T logits',28,True);text(1305,197,'z'+sup('(T)'),32)
    box('grassmann-branch',270,260,1120,450,'#f5faf6','#5aab7c')
    text(562,302,'Grassmann branch (G)',32,True)
    parts.append('<rect id="plucker-processing-group" x="645" y="315" width="725" height="203" rx="11" fill="none" stroke="#84bb92" stroke-width="1.4" stroke-dasharray="8 7"/>')
    text(1007,342,'2D subspace &amp; Plücker processing',28)
    modules=[('token-embedding-g',295,150,['Token','Embedding'],'#ffe8e9'),('linear-projection',475,150,['Linear','projection','W'+sub('red')],'#eff5f5'),('local-pairs',655,200,['Local pairs','(z'+sub('t')+', z'+sub('t−Δ')+')','z'+sub('t−Δ')+' → z'+sub('t'),'Δ ∈ {1, 2, 4}'],'#f3faff'),('plucker-coordinates',885,170,['Plücker','coordinates','p'+sub('ij')],'#f3faff')]
    for id,x,w,lines,fill in modules:
        box(id,x,356,w,150,fill,'#f17779' if id=='token-embedding-g' else '#368bea')
        ys={2:[411,447],3:[397,431,474],4:[390,425,461,494]}[len(lines)]
        for y,s in zip(ys,lines):text(x+w/2,y,s,32 if 'tspan' in s else 28,owner=f'{x+3},359,{w-6},144')
    box('normalization',1085,356,270,150,'#f3faff','#368bea')
    text(1220,391,'Normalization',28)
    text(1127,449,'p̂ =',32)
    text(1240,430,'p',32)
    parts.append('<path d="M1174 439H1333" stroke="#0b1c35" stroke-width="1.5"/>')
    text(1254,477,'max(‖p‖'+sub('2')+', ε)',32)
    for a,b in [(445,475),(625,655),(855,885),(1055,1085)]:arrow(f'M{a+1} 431H{b-2}')
    # The folded connector preserves the original one-direction data flow.
    arrow('M1220 507V545H385V578')
    for id,x,w,lines,fill in [('geometric-projection',295,180,['Geometric','projection'],'#f3faff'),('window-mean',510,170,['Window','mean','valid Δ'],'#ebf6ec'),('gated-mixing',715,170,['Gated','mixing'],'#ebf6ec'),('lm-head-g',920,170,['LM Head'],'#eee7fc'),('g-logits',1125,220,['G logits','z'+sup('(G)')],'#f3faf4')]:
        box(id,x,580,w,105,fill,'#a188e8' if id=='lm-head-g' else '#85b897')
        ys={1:[643],2:[620,658],3:[610,642,674]}[len(lines)]
        for y,s in zip(ys,lines):text(x+w/2,y,s,32 if 'tspan' in s else 28,owner=f'{x+3},583,{w-6},99')
    for a,b in [(475,510),(680,715),(885,920),(1090,1125)]:arrow(f'M{a+1} 631H{b-2}')
    arrow('M199 375H230V169H293');arrow('M230 375V431H293')
    box('late-fusion',1430,298,320,150,'#fff7df','#c9a843')
    text(1590,339,'Late logit fusion',30,True)
    text(1590,402,'z = αz'+sup('(T)')+' + (1 − α)z'+sup('(G)'),32,owner='1434,355,312,85')
    arrow('M1371 169H1410V375H1428');arrow('M1346 631H1410V375')
    box('teacher-logits',1465,500,250,112,'#f5f6f7','#919aa5')
    text(1590,543,'Teacher logits',28,True);text(1590,585,'z',32)
    arrow('M1590 449V498')
    parts.append('</svg>')
    return '\n'.join(parts)+'\n'


if __name__=='__main__':
    names=['fig1_paired_protocol','fig3_teacher_quality','fig4_ensemble_control']
    print(json.dumps({**{n:propose(n) for n in names},'figA1_teacher_architecture':architecture()}))
