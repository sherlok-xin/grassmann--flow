BUILD_DIR ?= ../../outputs/coling2027_arr_zh

.PHONY: all qa
all:
	mkdir -p "$(BUILD_DIR)"
	xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="$(BUILD_DIR)" main.tex
	cd "$(BUILD_DIR)" && BIBINPUTS="$(CURDIR):" BSTINPUTS="$(CURDIR):" bibtex main
	xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="$(BUILD_DIR)" main.tex
	xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="$(BUILD_DIR)" main.tex
	xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="$(BUILD_DIR)" main.tex

qa:
	python3 scripts/audit_chinese_archive.py
