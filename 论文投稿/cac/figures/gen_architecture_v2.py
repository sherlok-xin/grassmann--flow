#!/usr/bin/env python3
"""Draw the CAC architecture as editable SVG, vector PDF, and high-resolution PNG.

Requires CairoSVG for PDF/PNG export. The SVG uses native paths and text;
it does not embed or trace the generated raster concept.
"""
from pathlib import Path
from html import escape
import cairosvg
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parent
W, H = 1800, 1030
C = dict(ink="#46525F", line="#8996A2", faint="#DCE2E7", blue="#EAF1F8",
         blue_edge="#A7BBCD", green="#ECF4EE", green_edge="#ADC4B3",
         sand="#F8F0E2", sand_edge="#CDBE9E", lavender="#F0ECF7",
         lavender_edge="#BDB0CF", peach="#FAEFE7", neutral="#F6F8FA")
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="7.05in" '
         'height="4.0342in" viewBox="0 0 1800 1030" role="img" '
         'aria-labelledby="title desc">',
         '<title id="title">Grassmann–Transformer teacher–student architecture</title>',
         '<desc id="desc">Frozen late-fusion teacher and trainable hybrid student '
         'with alternative initialization, forward KL and next-token CE losses, '
         'and an inset showing causal Grassmann mixing with a hidden-state bypass.</desc>',
         '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
         'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M 0 0 L 10 5 L 0 10 z" fill="#8996A2"/></marker></defs>',
         '<rect width="1800" height="1030" fill="#FFFFFF"/>']


def box(x, y, w, h, fill="neutral", edge="faint", radius=10):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'rx="{radius}" fill="{C.get(fill, fill)}" '
                 f'stroke="{C.get(edge, edge)}" stroke-width="1.7"/>')


def text(x, y, value, size=25, anchor="middle", weight=400, color="ink", raw=False):
    if raw:
        formula = value.replace('∥', r'\Vert').replace('⊙', r'\odot')
        formula = formula.replace('−', '-').replace('λ', r'\lambda ').replace('Δ', r'\Delta ').replace('²', '^2')
        path = TextPath((0, 0), '$' + formula + '$', size=size,
                        prop=FontProperties(family='DejaVu Sans'))
        bounds = path.get_extents()
        shift = (bounds.x0 + bounds.x1)/2 if anchor == 'middle' else bounds.x0
        commands = []
        for verts, code in path.iter_segments():
            op = {1: 'M', 2: 'L', 3: 'Q', 4: 'C', 79: 'Z'}[code]
            commands.append(op if code == 79 else op + ' '.join(f'{v:.3f}' for v in verts))
        parts.append(f'<g transform="translate({x-shift:.3f} {y}) scale(1 -1)" '
                     f'fill="{C.get(color,color)}" aria-label="{escape(value)}">'
                     f'<path d="{" ".join(commands)}"/></g>')
        return
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                 f'font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" '
                 f'font-weight="{weight}" fill="{C.get(color, color)}">'
                 f'{value if raw else escape(value)}</text>')


def sub(base, index):
    return base + '_{' + index + '}'


def line(d, arrow=True, dashed=False, width=2):
    parts.append(f'<path d="{d}" fill="none" stroke="{C["line"]}" '
                 f'stroke-width="{width}" stroke-linejoin="round" '
                 f'stroke-linecap="round"'
                 + (' stroke-dasharray="6 6"' if dashed else '')
                 + (' marker-end="url(#arrow)"' if arrow else '') + '/>')


def dot(x, y):
    parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{C["line"]}"/>')


def lock(x, y):
    line(f'M {x+5} {y+11} V {y+5} C {x+5} {y-4} {x+19} {y-4} {x+19} {y+5} V {y+11}', False)
    box(x, y+9, 24, 20, "#FFFFFF", "line", 4)
    dot(x+12, y+18)


def branch_icon(x, y, grassmann=False):
    if grassmann:
        parts.append(f'<path d="M{x},{y+35} L{x+17},{y+3} L{x+51},{y+3} '
                     f'L{x+34},{y+35} Z" fill="#DCEADF" stroke="#ADC4B3" stroke-width="1.5"/>')
        line(f'M{x+12} {y+29} L{x+43} {y+9}')
        line(f'M{x+12} {y+29} L{x+39} {y+29}')
    else:
        for row in range(4):
            for col in range(4):
                box(x+col*12, y+row*10, 8, 7,
                    "#BDD0E1" if col<=row else "#F9FBFD", "#FFFFFF", 1)


text(30, 42, "(a)  Teacher–student distillation", 29, "start", 600)
text(580, 81, "CE pretraining → joint tuning", 21)

# The input is shared, but each model has its own two branch inputs.
box(25, 292, 155, 97, "neutral", "faint")
text(102, 324, "Input tokens", 24)
for i in range(5):
    box(40+i*27, 342, 21, 27, "blue", "blue_edge", 3)
    text(50+i*27, 361, str(i+1), 15)
line('M180 340 H213 V219 H269', False)
line('M213 340 V479 H269', False)
dot(213, 340)


def model(y, name, frozen):
    box(249, y, 819, 237, "#FFFFFF", "faint", 14)
    text(275, y+30, name, 29, "start", 600)
    if frozen:
        lock(430, y+9)
        box(468, y+6, 257, 33, "blue", "#FFFFFF", 7)
        text(596, y+30, "Frozen during KD", 21)
    else:
        box(520, y+6, 136, 33, "peach", "#FFFFFF", 7)
        text(588, y+30, "Trainable", 21)
    # Branch inputs connect separately and enter the actual blocks.
    line(f'M269 {y+124} V{y+82} H300')
    line(f'M269 {y+124} V{y+177} H300')
    dot(269, y+124)
    for offset, label, note, color, edge, is_gr in [
        (46, "Transformer", "Self-attention", "blue", "blue_edge", False),
        (141, "Grassmann", "Causal mixing", "green", "green_edge", True)]:
        box(300, y+offset, 235, 73, color, edge)
        text(321, y+offset+29, label, 22, "start", 500)
        text(321, y+offset+54, note, 19, "start")
        branch_icon(478, y+offset+18, is_gr)
    center=y+124
    box(601, center-49, 182, 98, "sand", "sand_edge")
    text(692, center-12, "Late fusion", 25, weight=500)
    text(692, center+17, "Weighted logits", 19)
    line(f'M535 {y+82} H570 V{center-21} H601')
    line(f'M535 {y+177} H570 V{center+21} H601')
    text(567, y+60, "logits", 17)
    text(567, y+210, "logits", 17)
    line(f'M783 {center} H824')
    box(824, center-28, 140, 56, "neutral", "faint")
    text(894, center+8, "Softmax / T", 21)
    line(f'M964 {center} H992')
    box(992, center-28, 59, 56, "#FFFFFF", "faint")
    text(1021, center+7, sub("p", "T" if frozen else "S"), 27, raw=True)
    return center


ty=model(95, "Teacher", True)
sy=model(355, "Hybrid student", False)

# Supervision: probabilities to KL, unsoftened student logits to CE.
line(f'M1051 {ty} H1239 V274')
text(1150, ty-15, sub("p", "T"), 24, raw=True)
line(f'M1051 {sy} H1112 V325 H1150')
text(1088, sy-16, sub("p", "S"), 24, raw=True)
box(1150, 274, 191, 100, "lavender", "lavender_edge")
text(1245, 307, "KD loss", 27, weight=500)
text(1245, 342, f'KL({sub("p", "T")} ∥ {sub("p", "S")})', 25, raw=True)
text(1245, 403, "T = 2", 21)

dot(802, sy)
line(f'M802 {sy} V553 H1180 V524')
text(968, 544, sub("z", "S"), 25, raw=True)
box(1150, 443, 191, 81, "neutral", "faint")
text(1245, 492, "CE loss", 27, weight=500)
box(1117, 591, 256, 47, "neutral", "faint")
text(1245, 622, "Next-token labels", 22)
line('M1245 591 V524')

line('M1341 324 H1385 V375 H1440')
line('M1341 482 H1385 V418 H1440')
box(1440, 329, 323, 133, "lavender", "lavender_edge")
text(1601, 362, "Training loss", 28, weight=500)
text(1601, 399, f'L = (1−λ){sub("L", "CE")}', 26, raw=True)
text(1601, 436, f'+ λT²{sub("L", "KD")}', 26, raw=True)
text(1601, 501, "Update student only", 22)

# Initialization is a parameter choice, represented by dashed arrows.
text(626, 613, "Initialization", 20)
box(406, 639, 180, 51, "peach", "sand_edge")
text(496, 672, "Random init.", 23)
text(627, 672, "or", 21)
box(668, 639, 239, 51, "green", "green_edge")
text(787, 672, "CE warm-start", 23)
line('M496 639 V592', dashed=True)
line('M787 639 V592', dashed=True)

# Grassmann inset, with an explicit hidden-state bypass.
line('M30 723 H1768', False, width=1)
text(30, 763, "(b)  Causal Grassmann mixing", 29, "start", 600)
text(49, 875, sub("h", "t"), 29, raw=True)
line('M78 867 H110')
stages=[(110, 155, "Linear", "reduction"),
        (305, 185, "Causal", "pairs"),
        (530, 177, "Plücker", "encoding"),
        (747, 120, "L2 norm", ""),
        (907, 186, "Project", "+ average"),
        (1133, 372, "Feature-wise gate", ""),
        (1545, 172, "LayerNorm", "")]
for x,w,a,b in stages:
    box(x, 798, w, 139, "green", "green_edge")
    text(x+w/2, 829, a, 23, weight=500)
    if b:
        text(x+w/2, 854, b, 23, weight=500)
for (x,w,_,_), (nx,_,_,_) in zip(stages, stages[1:]):
    line(f'M{x+w} 867 H{nx}')
text(187, 908, 'd → r', 26)
text(397, 885, f'{sub("z", "t")}, {sub("z", "t−Δ")}', 26, raw=True)
text(397, 916, 'Δ ∈ {1, 2, 4}', 20)
branch_icon(589, 874, True)
text(807, 890, 'p / ‖p‖₂', 22)
text(1000, 908, sub("g", "t"), 28, raw=True)
for i,fill in enumerate(["#DDEBE2", "#C7DCD0", "#E9F1EC", "#BCD3C5", "#D4E5DA"]):
    box(1227+i*33, 850, 24, 20, fill, "green_edge", 3)
text(1319, 910, f'{sub("a", "t")} ⊙ {sub("h", "t")} + (1−{sub("a", "t")}) ⊙ {sub("g", "t")}', 25, raw=True)
text(1631, 889, 'LN(·)', 28)
line('M1717 867 H1760')
text(1758, 911, sub("h", "out"), 25, raw=True)
dot(88, 867)
line('M88 867 V982 H1319 V937')
text(732, 973, sub("h", "t"), 25, raw=True)

parts.append('</svg>')
svg = '\n'.join(parts)
path = OUT / 'fig_architecture_v2.svg'
path.write_text(svg, encoding='utf-8')
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT/'fig_architecture_v2.pdf'))
cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT/'fig_architecture_v2.png'),
                output_width=3000, output_height=1717)
print(path)


if __name__ == '__main__':
    pass
