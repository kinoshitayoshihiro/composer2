demo:
	python tools/generate_demo_midis.py -m config/main_cfg.yml

demo-sax:
	python tools/generate_demo_midis.py -m config/sax_demo.yml --sections "Sax Solo"

test:
	pytest tests
