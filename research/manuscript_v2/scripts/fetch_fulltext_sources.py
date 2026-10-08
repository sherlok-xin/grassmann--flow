#!/usr/bin/env python3
"""Public-reference retrieval only. Never imports project models or experiments.

PDFs/text/raw exports are scratch build inputs under outputs, not git artifacts.
The author must read methods before recording a full-text review.
"""
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = ROOT / 'outputs/manuscript_v2_reference_review'
EXTRA = {
    'hinton2015distilling': ('https://doi.org/10.48550/arXiv.1503.02531', 'https://arxiv.org/pdf/1503.02531'),
    'busbridge2025scaling': ('https://proceedings.mlr.press/v267/busbridge25a.html', 'https://raw.githubusercontent.com/mlresearch/v267/main/assets/busbridge25a/busbridge25a.pdf'),
    'panigrahi2026grace': ('https://proceedings.iclr.cc/paper_files/paper/2026/hash/59fa5d049b777d880fdcddba2b24738f-Abstract-Conference.html', 'https://proceedings.iclr.cc/paper_files/paper/2026/file/59fa5d049b777d880fdcddba2b24738f-Paper-Conference.pdf'),
    'marcus1993ptb': ('https://aclanthology.org/J93-2004.bib', 'https://aclanthology.org/J93-2004.pdf'),
    'merity2016wikitext': ('https://doi.org/10.48550/arXiv.1609.07843', 'https://arxiv.org/pdf/1609.07843'),
    'eldan2023tinystories': ('https://doi.org/10.48550/arXiv.2305.07759', 'https://arxiv.org/pdf/2305.07759'),
    'vaswani2017attention': ('https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html', 'https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'),
    'benallal2025smollm2': ('https://doi.org/10.48550/arXiv.2502.02737', 'https://arxiv.org/pdf/2502.02737'),
    'penedo2024fineweb': ('https://proceedings.neurips.cc/paper_files/paper/2024/hash/370df50ccfdf8bde18f8f9c2d9151bda-Abstract-Datasets_and_Benchmarks_Track.html', 'https://proceedings.neurips.cc/paper_files/paper/2024/file/370df50ccfdf8bde18f8f9c2d9151bda-Paper-Datasets_and_Benchmarks_Track.pdf'),
    'edelman1998geometry': ('https://api.crossref.org/works/10.1137%2FS0895479895290954/transform/application/x-bibtex', 'https://arxiv.org/pdf/physics/9806030'),
}


def retrieve(record):
    key, metadata_url, pdf_url = record
    result = dict(key=key, metadata_url=metadata_url, pdf_url=pdf_url)
    try:
        r = requests.get(metadata_url, headers={'Accept': 'application/x-bibtex'} if 'doi.org' in metadata_url else {}, timeout=40)
        r.raise_for_status()
        r.encoding = 'utf-8'
        raw = r.text
        if '@' not in raw[:200] and '<html' in raw.lower():
            soup = BeautifulSoup(raw, 'html.parser')
            links = [urljoin(r.url, a['href']) for a in soup.find_all('a', href=True) if a.get_text(strip=True).lower() == 'bibtex']
            if links:
                metadata_url = links[0]
                r = requests.get(metadata_url, timeout=40)
                r.raise_for_status()
                r.encoding = 'utf-8'
                raw = r.text
            elif 'proceedings.mlr.press' in metadata_url:
                text = soup.get_text('\n')
                start = text.lower().index('@inproceedings')
                depth = 0
                opened = False
                for i in range(start, len(text)):
                    if text[i] == '{':
                        depth += 1
                        opened = True
                    elif text[i] == '}':
                        depth -= 1
                        if opened and depth == 0:
                            raw = text[start:i+1]
                            break
        if not raw.lstrip().startswith('@'):
            raise ValueError('No exported BibTeX; no fabricated fallback')
        (SCRATCH / f'{key}.bib').write_text(raw)
        result.update(export_url=metadata_url, export_sha256=hashlib.sha256(raw.encode()).hexdigest(), metadata_status='EXPORTED_PENDING_REVIEW')
    except Exception as e:
        result.update(metadata_status='VERIFY_EXTERNAL', metadata_error=str(e))
    if pdf_url:
        try:
            r = requests.get(pdf_url, timeout=90)
            r.raise_for_status()
            if not r.content.startswith(b'%PDF'):
                raise ValueError('Not a PDF')
            pdf = SCRATCH / f'{key}.pdf'
            pdf.write_bytes(r.content)
            subprocess.run(['pdftotext', '-layout', str(pdf), str(pdf.with_suffix('.txt'))], check=True)
            result.update(pdf_sha256=hashlib.sha256(r.content).hexdigest(), fulltext_status='DOWNLOADED_NOT_YET_REVIEWED')
        except Exception as e:
            result.update(fulltext_status='FETCH_FAILED', fulltext_error=str(e))
    return result


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / 'research/manuscript_v2/citation_verification.json').read_text())['records']
    records = []
    for r in old:
        url = r['primary_url']
        if not '.pdf' in url:
            fixed = {
                'cho2019efficacy': 'https://openaccess.thecvf.com/content_ICCV_2019/papers/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.pdf',
                'yuan2021selection': 'https://arxiv.org/pdf/2012.06048',
                'menon2021statistical': 'https://proceedings.mlr.press/v139/menon21a/menon21a.pdf',
                'xie2026adakd': 'https://ojs.aaai.org/index.php/AAAI/article/view/40701/44662',
                'jin2026entropy': 'https://raw.githubusercontent.com/mlresearch/v306/main/assets/jin26e/jin26e.pdf',
            }
            try:
                page = requests.get(url, timeout=15)
                soup = BeautifulSoup(page.text, 'html.parser')
                candidates = [urljoin(url, a['href']) for a in soup.find_all('a', href=True) if ('.pdf' in a['href'] and (a.get_text(strip=True).lower() in ('pdf', 'paper') or 'pdf' in a.get_text(strip=True).lower()))]
            except requests.RequestException:
                candidates = []
            if r['key'] in fixed:
                candidates = [fixed[r['key']]]
            if candidates:
                url = candidates[0]
            elif 'aclanthology.org' in url:
                url = url.rstrip('/') + '.pdf'
        records.append((r['key'], r['export_url'], url))
    records += [(k, *v) for k, v in EXTRA.items()]
    results = list(ThreadPoolExecutor(max_workers=8).map(retrieve, records))
    (SCRATCH / 'retrieval.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
