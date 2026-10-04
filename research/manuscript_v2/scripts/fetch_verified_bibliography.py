#!/usr/bin/env python3
"""Fetch publisher-exported metadata only; never synthesize unverified citations.

Local bibliography maintenance, not an experiment. Requires requests/bs4.
Failed records are explicitly excluded and marked VERIFY_EXTERNAL.
"""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/manuscript_v2"
MANUSCRIPT = ROOT / "论文投稿/manuscript_v2"
DATE = "2026-10-04"

# URLs below were inspected on official proceedings/Anthology/publisher pages.
RECORDS = [
    ("cho2019efficacy", "On the Efficacy of Knowledge Distillation", 2019, "ICCV",
     "https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html", "html", "arXiv:1910.01348"),
    ("park2021friendly", "Learning Student-Friendly Teacher Networks for Knowledge Distillation", 2021, "NeurIPS",
     "https://proceedings.neurips.cc/paper_files/paper/12641-/bibtex", "bib", "arXiv:2102.07650"),
    ("yuan2021selection", "Reinforced Multi-Teacher Selection for Knowledge Distillation", 2021, "AAAI",
     "https://doi.org/10.1609/aaai.v35i16.17680", "doi", "DOI:10.1609/aaai.v35i16.17680; arXiv:2012.06048"),
    ("zhong2024atkd", "Revisiting Knowledge Distillation for Autoregressive Language Models", 2024, "ACL",
     "https://aclanthology.org/2024.acl-long.587.bib", "bib", "ACL:2024.acl-long.587; arXiv:2402.11890"),
    ("stanton2021really", "Does Knowledge Distillation Really Work?", 2021, "NeurIPS",
     "https://proceedings.neurips.cc/paper_files/paper/12152-/bibtex", "bib", "arXiv:2106.05945; OpenReview:7J-fKoXiReA"),
    ("du2020agree", "Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space", 2020, "NeurIPS",
     "https://proceedings.neurips.cc/paper_files/paper/10759-/bibtex", "bib", "NeurIPS:91c77393975889bd08f301c9e13a44b7"),
    ("zhou2022metadistil", "BERT Learns to Teach: Knowledge Distillation with Meta Learning", 2022, "ACL",
     "https://aclanthology.org/2022.acl-long.485.bib", "bib", "ACL:2022.acl-long.485; arXiv:2106.04570"),
    ("kaplun2022bad", "Knowledge Distillation: Bad Models Can Be Good Role Models", 2022, "NeurIPS",
     "https://proceedings.neurips.cc/paper_files/paper/19353-/bibtex", "bib", "arXiv:2203.14649; NeurIPS:b88edf805e96654a4f9e7b783e854ae3"),
    ("menon2021statistical", "A statistical perspective on distillation", 2021, "ICML",
     "https://proceedings.mlr.press/v139/menon21a.html", "html", "PMLR:v139-menon21a; arXiv:2005.10419 (different preprint title)"),
    ("wang2021selective", "Selective Knowledge Distillation for Neural Machine Translation", 2021, "ACL-IJCNLP",
     "https://aclanthology.org/2021.acl-long.504.bib", "bib", "ACL:2021.acl-long.504; arXiv:2105.12967"),
    ("gu2024minillm", "MiniLLM: Knowledge Distillation of Large Language Models", 2024, "ICLR",
     "https://proceedings.iclr.cc/paper_files/paper/4166-/bibtex", "bib", "arXiv:2306.08543; ICLR:8ac015d409635f196f9e3e9dcfb9a94e"),
    ("agarwal2024gkd", "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes", 2024, "ICLR",
     "https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html", "linked", "OpenReview:3zKtaqxLhW; arXiv:2306.13649"),
    ("xie2026adakd", "LLM-Oriented Token-Adaptive Knowledge Distillation", 2026, "AAAI",
     "https://doi.org/10.1609/aaai.v40i40.40701", "doi", "DOI:10.1609/aaai.v40i40.40701"),
    ("jin2026entropy", "Entropy-Aware On-Policy Distillation of Language Models", 2026, "ICML",
     "https://proceedings.mlr.press/v306/jin26e.html", "html", "PMLR:v306-jin26e; arXiv:2603.07079"),
]

# These are bounded comparisons to verified abstracts, not claims of exhaustive
# full-text review or novelty relative to every prior protocol.
NOTES = {
    "cho2019efficacy": ("Teacher quality / compatibility", "More accurate teachers need not yield better distilled students; teacher capacity and training duration matter.", "We pair CE and KD from identical warm-start LM states and triangulate teacher-state, source and ensemble controls. Accuracy insufficiency itself is not new."),
    "park2021friendly": ("Student-oriented teacher training", "Student-friendly teacher representations are explicitly trained with student branches.", "Our teacher-state intervention changes frozen supervision while holding composition fixed; no new student-friendly teacher-training algorithm is proposed."),
    "yuan2021selection": ("Teacher selection / multi-teacher compatibility", "Instance-dependent teacher selection accounts for differing teacher suitability and student capacity.", "Our source and ensemble contrasts keep supervision rules fixed rather than optimizing a teacher-selection policy. We do not benchmark that policy."),
    "zhong2024atkd": ("Autoregressive adaptive KD; closest direct prior", "Larger autoregressive teachers can yield poorer students; token teaching modes motivate ATKD.", "The contribution is a matched warm-start CE-relative evidence chain, including independent student seeds, fixed-composition state changes and a homogeneous ensemble falsification. No ATKD baseline was run; do not claim superiority or first discovery of harmful LM KD."),
    "stanton2021really": ("Ensemble KD / optimization", "Student imitation fidelity and generalization differ; optimization can impede teacher matching.", "Our endpoint is improvement over matched CE rather than fidelity. The bounded branch/ensemble and continuation-regime controls are empirical evidence, not a new account of optimization."),
    "du2020agree": ("Gradient-aware ensemble distillation", "Gradient-space teacher agreement/diversity informs adaptive ensemble supervision.", "We test fixed teacher composition and offline local-gradient endpoint diagnostics. Their failure does not refute online adaptive weighting or the full method of this work."),
    "zhou2022metadistil": ("Validation-driven / student-oriented distillation", "Student feedback and meta learning improve the suitability of teacher supervision.", "Our fixed teacher-state intervention and failed offline prediction are not meta-optimization. We do not test or invalidate MetaDistil."),
    "kaplun2022bad": ("Weak teacher / limits of teacher quality", "A poor classifier can nevertheless provide useful distillation targets under the studied theoretical setting.", "We give a held-out-NLL counterexample in warm-start autoregressive modeling, not a new weak-teacher principle or extension of the classifier theorem."),
    "menon2021statistical": ("Teacher quality / statistical explanation", "Probability estimation and bias/variance can explain distillation benefits beyond classifier accuracy.", "Our scalar-likelihood counterexample does not imply teacher quality is irrelevant; we supply controlled LM endpoints rather than a new statistical theory."),
    "wang2021selective": ("Harmful supervision / negative transfer", "Teacher supervision on some training samples can hurt NMT; selective distillation addresses this.", "We use frozen token-mean forward KL and paired continuation outcomes, without sample selection. Harmful supervision and selection are established ideas, not novel claims here."),
    "gu2024minillm": ("Modern autoregressive KD / loss choice", "Reverse-KL distillation addresses distributional properties of large autoregressive teachers and students.", "Our forward-KL NLL protocol is not an evaluation of MiniLLM or instruction-following generation. Phase 4 cannot establish failure of modern KD in general."),
    "agarwal2024gkd": ("Modern autoregressive KD / student distribution", "Student-generated sequences and flexible divergence address train/inference distribution mismatch.", "We evaluate fixed-corpus next-token likelihood with a fixed objective, not on-policy generation or RL fine-tuning; no comparison against GKD was executed."),
    "xie2026adakd": ("Recent token-adaptive / student-state KD", "Token difficulty and dynamic student learning state guide adaptive loss focusing and temperature.", "The present synthesis does not newly introduce student-state dependence and contains no adaptive token method. Exact protocol overlap still requires full-text checking before a novelty claim."),
    "jin2026entropy": ("Recent uncertainty-aware on-policy KD", "Teacher entropy informs a mixture of forward and reverse KL to maintain diversity and learning signals.", "We study fixed forward KL and corpus likelihood, not on-policy reasoning or adaptive divergence. Diversity matters in prior algorithms; our bounded JSD result is not a general rejection of it."),
}


def parse_fields(raw):
    """Parse one publisher BibTeX entry, including quoted/nested values."""
    match = re.search(r"@(\w+)\s*\{([^,]+),", raw)
    if not match:
        raise ValueError("no publisher BibTeX entry")
    pos, fields = match.end(), {}
    while pos < len(raw):
        field = re.match(r"\s*,?\s*(\w+)\s*=\s*", raw[pos:])
        if not field:
            break
        name = field.group(1).lower()
        pos += field.end()
        start = pos
        if raw[pos] in '{"':
            opener = raw[pos]
            pos += 1
            depth = 1 if opener == "{" else 0
            value_start = pos
            while pos < len(raw):
                char = raw[pos]
                escaped = pos > 0 and raw[pos - 1] == "\\"
                if not escaped:
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if opener == "{" and depth == 0:
                            break
                    elif opener == '"' and char == '"' and depth == 0:
                        break
                pos += 1
            value = raw[value_start:pos]
            pos += 1
        else:
            while pos < len(raw) and raw[pos] not in ",}\n":
                pos += 1
            value = raw[start:pos].strip()
        fields[name] = re.sub(r"\s+", " ", value).strip()
    return match.group(1).lower(), fields


def fetch(record):
    key, expected, year, venue, url, mode, identifier = record
    audit = dict(key=key, expected_title=expected, expected_year=year,
                 venue=venue, identifier=identifier, export_url=url,
                 checked_on=DATE, reading_scope="official metadata and abstract; not an exhaustive full-text review")
    try:
        headers = {"Accept": "application/x-bibtex"} if mode == "doi" else {}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = "utf-8"
        raw = response.text
        if mode == "linked":
            soup = BeautifulSoup(raw, "html.parser")
            link = next(a["href"] for a in soup.find_all("a", href=True) if a.get_text(strip=True).lower() == "bibtex")
            from urllib.parse import urljoin
            url = urljoin(response.url, link)
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            response.encoding = "utf-8"
            raw = response.text
            audit["export_url"] = url
        elif mode == "html":
            text = BeautifulSoup(raw, "html.parser").get_text("\n")
            start = re.search(r"@InProceedings\s*\{", text, re.I)
            if not start:
                raise ValueError("official page contains no exported BibTeX")
            depth, end = 0, start.end() - 1
            for end in range(end, len(text)):
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                    if depth == 0:
                        break
            raw = text[start.start():end + 1]
        kind, fields = parse_fields(raw)
        normalized = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        if normalized(fields.get("title", "")) != normalized(expected):
            raise ValueError("exported title differs from inspected record")
        if fields.get("year") != str(year) or not fields.get("author"):
            raise ValueError("year/author mismatch or missing metadata")
        if not (fields.get("booktitle") or fields.get("journal")):
            raise ValueError("missing venue")
        if key == "park2021friendly":
            # Official export lowercases Jeong and joins the given name of Kim.
            # The primary proceedings PDF title page was inspected 2026-10-04.
            audit["author_name_normalization"] = dict(
                reason="case and given-name spacing verified against PDF title page",
                source="https://proceedings.neurips.cc/paper/2021/file/6e7d2da6d3953058db75714ac400b584-Paper.pdf",
                original_author=fields["author"])
            fields["author"] = fields["author"].replace("jeong, changwook", "Jeong, Changwook").replace("Kim, Daesin", "Kim, Dae Sin")
        audit.update(status="VERIFIED_METADATA", title=fields["title"], authors=fields["author"],
                     year=year, doi=fields.get("doi", ""), primary_url=fields.get("url", url),
                     original_export_sha256=hashlib.sha256(raw.encode()).hexdigest())
        allowed = ["title", "author", "booktitle", "journal", "year", "volume", "number", "pages", "publisher", "doi", "url"]
        lines = [f"@{kind}{{{key},"]
        for name in allowed:
            if name in fields:
                value = fields[name].replace("–", "--").replace("—", "--")
                if name == "title":
                    value = "{" + value + "}"  # Protect verified title casing.
                lines.append(f"  {name} = {{{value}}},")
        lines.append("}")
        return audit, "\n".join(lines)
    except Exception as exc:
        audit.update(status="VERIFY_EXTERNAL", error=str(exc))
        return audit, ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    results = list(ThreadPoolExecutor(max_workers=6).map(fetch, RECORDS))
    records = [audit for audit, _ in results]
    (OUT / "citation_verification.json").write_text(json.dumps(dict(checked_on=DATE, records=records), indent=2) + "\n")
    bib = "% Publisher-exported metadata; verified 2026-10-04. No unverified entries.\n\n"
    bib += "\n\n".join(entry for _, entry in results if entry) + "\n"
    (MANUSCRIPT / "references.bib").write_text(bib)
    intro = """# Related-work overlap audit (V2)

Checked on 2026-10-04. This is a focused overlap audit, not a systematic review,
and not evidence of being the first work on any concept. Searches covered
official CVF, NeurIPS, ACL Anthology, AAAI, ICLR/OpenReview, PMLR and arXiv.
Queries included teacher quality/student-friendly teaching; teacher selection
and compatibility; ATKD autoregressive adaptive distillation; ensemble KD;
gradient-space/validation-driven KD; harmful supervision and weak teachers;
and recent token/entropy-adaptive LM distillation. Abstracts and publisher
metadata were checked; complete method/protocol comparisons remain writing
TODOs. No adaptive method listed below was evaluated in our experiments.

Publisher BibTeX was fetched, title/year/authors/venue presence checked and
reduced mechanically to compact metadata. One exported author-case/given-name
spacing issue was corrected against the official PDF title page and recorded.
`citation_verification.json`
records the export URL, identifier, metadata scope and source hash. A failed
fetch is `VERIFY_EXTERNAL` and is excluded from `references.bib`; it is not
quietly replaced by a hand-invented citation. Abstract text is not reproduced.

## Positioning by area

Teacher-quality and student-oriented training: Cho/Hariharan, Park et al.,
Menon et al. and Kaplun et al. already undermine simple teacher-accuracy rules.
Teacher selection/compatibility: Yuan et al. and MetaDistil already adapt
supervision to the student. Autoregressive adaptive KD: ATKD is the closest
direct antecedent for harmful LM KD; MiniLLM/GKD/AdaKD show that the objective
and student distribution/state are established concerns. Ensemble/multi-teacher:
Stanton et al. and Du et al. make diversity, fidelity and optimization relevant.
Gradient/validation-driven: Du et al. and MetaDistil prevent any novelty claim
for gradient or held-out feedback alone. Harmful supervision: Wang et al. and
ATKD directly precede our negative-transfer observations. Recent AdaKD and
entropy-aware on-policy work must not be omitted from a final 2026 literature
pass. Our failed offline diagnostic does not invalidate those adaptive methods.

## Verified records and bounded distinctions
"""
    parts = [intro]
    for audit in records:
        area, overlap, distinction = NOTES[audit["key"]]
        parts += [f"\n### {audit['key']}\n",
                  f"Title: {audit.get('title', audit['expected_title'])}.\n\n",
                  f"Authors: {audit.get('authors', 'VERIFY_EXTERNAL')}.\n\n",
                  f"Year / venue: {audit['expected_year']} / {audit['venue']}. "
                  f"Identifiers: {audit['identifier']}" + (f"; DOI:{audit['doi']}" if audit.get("doi") else "") + ".\n\n",
                  f"Status: {audit['status']}. Area: {area}. "
                  f"[Primary metadata/export]({audit['export_url']})" +
                  (f"; [publisher paper/page]({audit['primary_url']})" if audit.get("primary_url") else "") + ".\n\n",
                  f"Exact overlap: {overlap}\n\nExact remaining distinction: {distinction}\n"]
    parts.append("""
## Remaining gaps before prose is finalized

Read the closest full papers, especially ATKD, MetaDistil, MiniLLM/GKD and
recent AdaKD, before asserting differences in initialization, CE comparators,
checkpoint selection or training budgets. The current abstract-level audit
does not prove that any prior work lacks a warm-start control. Do not claim
state-of-the-art KD, superiority over adaptive KD, or a new general theory.
The defensible position is the specific paired, multi-seed intervention and
falsification evidence chain assembled here. It is a bounded empirical study.

Dataset/model/architecture citations still need primary-source verification
before full manuscript writing (PTB, WT2, TinyStories, FineWeb-Edu, SmolLM2,
Transformer, Grassmann flow/Pluecker background and foundational KD).
These are `VERIFY_EXTERNAL` writing TODOs, not entries copied from the old
manuscript. A venue is not specified: `criteria_binding_unavailable`.
Do a focused literature refresh at the actual writing/submission date,
without launching experiments or implying a complete literature census.
""")
    (OUT / "RELATED_WORK_GAPS.md").write_text("".join(parts))
    print(json.dumps({"verified": sum(x["status"] == "VERIFIED_METADATA" for x in records),
                      "pending": [x["key"] for x in records if x["status"] != "VERIFIED_METADATA"]}))


if __name__ == "__main__":
    main()
