demo:
	python tools/generate_demo_midis.py -m config/main_cfg.yml

demo-sax:
	python tools/generate_demo_midis.py -m config/sax_demo.yml --sections "Sax Solo"

test:
	pytest tests

test-controls:
	pytest tests/test_controls_spline.py tests/test_apply_controls.py -q

dev:
	python -m venv .venv && \
	. .venv/bin/activate && \
	pip install -r requirements.txt && \
	pip install -e .[test] && \
	coverage run -m pytest -q
