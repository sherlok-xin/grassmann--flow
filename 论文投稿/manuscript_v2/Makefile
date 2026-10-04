BUILD_DIR := ../../outputs/manuscript_v2_build

.PHONY: all
all:
	mkdir -p $(BUILD_DIR)
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) main.tex
