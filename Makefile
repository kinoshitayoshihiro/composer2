demo:
	python tools/generate_demo_midis.py -m config/main_cfg.yml

demo-sax:
	python tools/generate_demo_midis.py -m config/main_cfg.yml --sections sax_solo

test:
	pytest tests
