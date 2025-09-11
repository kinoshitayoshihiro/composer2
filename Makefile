demo:
	python tools/generate_demo_midis.py -m config/main_cfg.yml

demo-sax:
	python tools/generate_demo_midis.py -m config/sax_demo.yml --sections "Sax Solo"

test:
	pytest tests

test-controls:
	pytest tests/test_controls_spline.py tests/test_apply_controls.py -q

format:
	black .

dev:
	python -m venv .venv && \
	. .venv/bin/activate && \
	pip install -r requirements.txt && \
	pip install -e .[test] && \
	coverage run -m pytest -q

# Phase 1 helpers -------------------------------------------------------------

normalize-midi:
	python -m scripts.normalize_midi_meta --in data/songs --out-dir data/songs_norm

phrase-predict-prod:
	python -m scripts.predict_phrase \
	  --ckpt checkpoints/guitar_lead/gtr_bilstm_tcn_crf_gap10.best.ckpt \
	  --in-midi data/songs_norm \
	  --out-csv checkpoints/guitar_lead/preds/gtr_songs_preds_final.csv \
	  --instrument-regex '(?i)guitar|DL_Guitar|ギター' --pitch-range 52 88 \
	  --no-path-hint \
	  --val-csv data/phrase_csv/gtr_midi_gap10_valid_midistem.csv \
	  --scan-range 0.10 0.90 0.01 \
	  --auto-th --per-song-th \
	  --min-gap 2 \
	  --device mps \
	  --report-json checkpoints/guitar_lead/preds/gtr_songs_preds_final.report.json

test-pb:
	pytest -q tests/test_pb_math.py

validate-demo-yaml:
	python - <<'PY'
	import yaml, jsonschema, pathlib
	root = pathlib.Path('data/demo_ballad')
	with open('conf/schemas/sections.schema.yaml') as f:
	    sec_schema = yaml.safe_load(f)
	with open('conf/schemas/emotion_profile.schema.yaml') as f:
	    emo_schema = yaml.safe_load(f)
	with open(root / 'sections.yaml') as f:
	    sec = yaml.safe_load(f)
	with open(root / 'emotion_profile.yaml') as f:
	    emo = yaml.safe_load(f)
	jsonschema.validate(instance=sec, schema=sec_schema)
	jsonschema.validate(instance=emo, schema=emo_schema)
	print('YAML validation OK')
	PY

train-pedal:
	python -m scripts.train_pedal \
	  data.train=data/pedal/train.csv \
	  data.val=data/pedal/val.csv \
	  trainer.max_epochs=15 \
	  trainer.accelerator=auto \
	  batch_size=32 \
	  learning_rate=1e-3

eval-pedal:
	python -m scripts.eval_pedal \
	  --csv data/pedal/val.csv \
	  --ckpt checkpoints/pedal.ckpt \
	  --window 64 --hop 16 \
	  --device auto

predict-pedal:
	python -m scripts.predict_pedal \
	  --in data/songs_norm \
	  --ckpt checkpoints/pedal.ckpt \
	  --out-dir outputs/pedal_pred \
	  --window 64 --hop 16 --batch 64

pedal-make-filelist:
	mkdir -p tmp && find data/songs_norm -type f \( -name '*.mid' -o -name '*.midi' \) > tmp/pedal_files.txt && wc -l tmp/pedal_files.txt

pedal-split-shards:
	mkdir -p tmp/pedal_shards && python -m scripts.pedal_split_filelist --file-list tmp/pedal_files.txt --out-dir tmp/pedal_shards --shards 16 --prefix list_

retry-pedal-fail:
	python -m scripts.retry_failed_pedal_frames \
	  --fail-list tmp/pedal_failed.txt \
	  --out-csv data/pedal/all.csv \
	  --sr 16000 --hop 4096 --cc-th 1 --max-seconds 60

pedal-merge-shards:
	python -m scripts.merge_csv_shards --glob 'data/pedal/frames_list_*' --out data/pedal/all_sharded.csv

pedal-frames-to-spans:
	python -m scripts.pedal_frames_to_spans --in data/pedal/all_sharded.csv --out data/pedal/all_spans.csv
