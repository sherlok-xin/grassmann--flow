# COLING 2027 / ARR 稿件中文版内部归档

本目录是最终英文稿的完整中文内部归档版，不是会议投稿文件，也不是摘要或报告改写。

权威英文来源提交：`5955954eed279686be9f4b1ba6634967d7676325`。
英文来源目录：`论文投稿/coling2027_arr/`。
英文投稿 PDF：`outputs/coling2027_arr/main.pdf`。
英文 PDF SHA256：`700490be24483fef4bfd69a367ade958981bbe7e05e4804c3c14efa40cde7255`。

中文源文件使用 XeLaTeX / ctexart，A4 单栏、11pt、25mm 页边距。
中文正文字体为 Noto Serif CJK SC，中文无衬线和等宽字体分别为 Noto Sans CJK SC 和 Noto Sans Mono CJK SC；西文字体为 TeX Gyre Termes / Heros / Cursor。
中文标题、章节、正文、图注、表头和附录文字均为译文；五张冻结矢量图保留英文图内文字。公式、实验数值、警告标识、引用键、标签和参考文献不改变。
`references.bib`、`acl_natbib.bst` 和五张图的 PDF 均为英文来源的独立副本，不使用符号链接。

从仓库根目录构建：

```bash
make -C 论文投稿/coling2027_arr_zh
```

输出：`outputs/coling2027_arr_zh/main.pdf`。默认构建仅写入中文版输出目录，不重新构建英文投稿稿件。
需要 XeLaTeX、BibTeX、ctex 和上述字体。参考文献沿用英文稿的 ACL author–year 样式，但不加载匿名评审版 ACL 模板或行号。

只读一致性检查：

```bash
make -C 论文投稿/coling2027_arr_zh qa
```

完整检查记录见 `research/manuscript_v2/COLING2027_CHINESE_ARCHIVE_STATUS.md`。
一致性检查需要 Python 3 和 PyMuPDF；它只读取源码与已构建 PDF，不运行模型或重算实验终点。

说明：本中文版仅用于课题组内部留档与报备。正式投稿版本为英文版 COLING 2027 / ACL Rolling Review 稿件。如中英文表述存在差异，以最终英文投稿稿为准。
