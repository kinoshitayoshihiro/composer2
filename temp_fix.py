import pathlib
files = [
    'tests/test_external_sync.py',
    'tests/test_live_buffer_integration.py',
    'tests/test_cli_hyperopt.py',
]
for f in files:
    p = pathlib.Path(f)
    try:
        txt = p.read_text()
    except FileNotFoundError:
        continue
    if 'import types' not in txt:
        p.write_text('import types\n' + txt)
print("done")
