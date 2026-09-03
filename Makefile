.PHONY: install test synthetic smoke tables figures clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

# A small synthetic cohort and a cut down search, enough to prove the pipeline
# runs end to end. It will not reproduce any published number.
synthetic:
	python synthetic/make_synthetic_cohort.py --scale 0.2

smoke: synthetic
	python scripts/01_build_matrices.py \
		--women work/synthetic/synthetic_women.csv \
		--men work/synthetic/synthetic_men.csv \
		--outdir work/results
	python scripts/02_run_benchmark.py --fresh --no-save-models \
		--n-iter 2 --bootstrap 50 --outdir work/results

tables:
	python scripts/04_make_tables.py

figures:
	python scripts/05_make_figures.py

clean:
	rm -rf work/ tables/*.csv figures/*.png figures/*.pdf
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
