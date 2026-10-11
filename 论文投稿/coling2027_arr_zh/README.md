# COLING 2027 / ARR 稿件中文 ACL 双栏版本

本目录复用最终英文稿的 ACL 双栏版式与完整中文译文，不是摘要或报告改写。

权威英文来源提交：`5955954eed279686be9f4b1ba6634967d7676325`。
英文来源目录：`论文投稿/coling2027_arr/`。
英文投稿 PDF：`outputs/coling2027_arr/main.pdf`。
英文 PDF SHA256：`700490be24483fef4bfd69a367ade958981bbe7e05e4804c3c14efa40cde7255`。

中文源文件使用 XeLaTeX、article 和非 review 模式的原版 `acl.sty`，由 ACL 模板设置双栏、页面尺寸、页边距及正文/图注字号。使用 fontspec / xeCJK 支持中文，不另设 geometry 或报告式版面。
中文正文字体为 Noto Serif CJK SC，中文无衬线和等宽字体分别为 Noto Sans CJK SC 和 Noto Sans Mono CJK SC；西文字体为 TeX Gyre Termes / Heros / Inconsolata。仅使用已安装字体，不附带字体文件。
中文标题、章节、正文、图注、表头和附录文字均为译文；五张冻结矢量图保留英文图内文字。公式、实验数值、警告标识、引用键、标签和参考文献不改变。
`acl.sty`、`references.bib`、`acl_natbib.bst` 和五张图的 PDF 均为英文来源的独立副本，不使用符号链接，也不修改模板样式。保留英文 figure* / table*、图表宽度、参考文献与附录的先后顺序及附录图 A1 编号。

从仓库根目录构建：

```bash
make -C 论文投稿/coling2027_arr_zh
```

输出：`outputs/coling2027_arr_zh/main.pdf`。默认构建仅写入中文版输出目录，不重新构建英文投稿稿件。
需要 XeLaTeX、BibTeX、fontspec、xeCJK 和上述字体。参考文献沿用英文稿的 ACL author–year 样式，非 review 模式显示作者占位模板，不显示匿名文字或评审行号。作者一为单位 1，作者二为单位 1,2，真实作者信息需手动填写。

只读一致性检查：

```bash
make -C 论文投稿/coling2027_arr_zh qa
```

完整检查记录见 `research/manuscript_v2/COLING2027_CHINESE_ARCHIVE_STATUS.md`。
一致性检查需要 Python 3 和 PyMuPDF；它只读取源码与已构建 PDF，不运行模型或重算实验终点。
