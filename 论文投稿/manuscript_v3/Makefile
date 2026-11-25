BUILD_DIR ?= ../../outputs/manuscript_v3

.PHONY: all
all:
	mkdir -p "$(BUILD_DIR)"
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$(BUILD_DIR)" main.tex
	cp "$(BUILD_DIR)/main.pdf" main.pdf
