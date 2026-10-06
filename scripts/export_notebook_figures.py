"""Export saved notebook PNG outputs without rerunning model training.

Run from any directory: python scripts/export_notebook_figures.py
Archives saved outputs, including historical variants; it does not regenerate absent figures.
"""
from pathlib import Path
import base64
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/notebook_figures'
OUT.mkdir(parents=True, exist_ok=True)
records, inventory = [], []
for notebook in sorted((ROOT / 'notebooks').glob('*.ipynb')):
    data = json.loads(notebook.read_text())
    heading, count = 'Notebook output', 0
    folder = OUT / notebook.stem
    for cell_no, cell in enumerate(data['cells'], 1):
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'markdown':
            headings = re.findall(r'^#{1,6}\s+(.+)', source, re.M)
            if headings:
                heading = headings[0]
        png_outputs = [o['data']['image/png'] for o in cell.get('outputs', []) if 'image/png' in o.get('data', {})]
        for output_no, encoded in enumerate(png_outputs, 1):
            slug = re.sub(r'[^a-z0-9]+', '-', heading.lower()).strip('-')[:65] or 'figure'
            name = f'cell-{cell_no:03d}-figure-{output_no:02d}-{slug}.png'
            folder.mkdir(parents=True, exist_ok=True)
            raw = base64.b64decode(''.join(encoded) if isinstance(encoded, list) else encoded)
            assert raw.startswith(b'\x89PNG\r\n\x1a\n'), notebook
            (folder / name).write_bytes(raw)
            records.append({'notebook': str(notebook.relative_to(ROOT)), 'cell_number': cell_no,
                'figure_in_cell': output_no, 'section': heading, 'file': str((folder / name).relative_to(ROOT)),
                'sha256': hashlib.sha256(raw).hexdigest()})
            count += 1
    inventory.append((notebook.stem, count))
lines = ['# Notebook figure archive', '',
    'PNG figures exported from saved notebook outputs without training or rerunning historical experiments.',
    'Notebook cells are numbered from one. Section names identify the source context, not an independently inferred plot title.',
    'Historical variants retain their historical labels and results; the completed reduced-schedule path and Code 012 are the current extension analyses.', '',
    '## Notebook inventory', '', '| Notebook | Saved figures |', '|---|---:|']
for stem, count in inventory:
    link = f'[{stem}]({stem}/)' if count else stem
    lines.append(f'| {link} | {count} |')
lines += ['', 'Zero means no embedded PNG figure was present in the saved notebook, not that a plot was regenerated or a full-schedule run completed.', '',
    '## Figure index', '', '| Notebook | Cell | Context | PNG |', '|---|---:|---|---|']
for r in records:
    relative = Path(r['file']).relative_to(OUT.relative_to(ROOT)).as_posix()
    lines.append(f"| {Path(r['notebook']).stem} | {r['cell_number']} | {r['section'].replace('|', '/')} | [Figure]({relative}) |")
(OUT / 'README.md').write_text('\n'.join(lines) + '\n')
(OUT / 'manifest.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
print(f'Exported {len(records)} figures from {len(inventory)} notebooks to {OUT}')
