"""Render executed Code 012 outputs as ordinary notebook cells, without output scrolling.
Run after reexecuting Code 012 to refresh its inline result blocks.
"""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'notebooks/012_bootstrap_sharpe_uncertainty.ipynb'

def main():
    notebook = json.loads(PATH.read_text())
    cells = [c for c in notebook['cells'] if not c.get('metadata', {}).get('code012_inline_result')]
    expanded = []
    count = 0
    for cell in cells:
        expanded.append(cell)
        if cell['cell_type'] != 'code' or cell.get('id') == 'expanded-results-display':
            continue
        outputs = cell.get('outputs', [])
        if not outputs:
            continue
        cell.setdefault('metadata', {}).update(scrolled=False, collapsed=True)
        cell['metadata'].setdefault('jupyter', {})['outputs_hidden'] = True
        for index, output in enumerate(outputs):
            data = output.get('data', {})
            attachments = {}
            if 'image/png' in data:
                name = 'result.png'
                attachments[name] = {'image/png': data['image/png']}
                source = '![Saved result figure](attachment:result.png)'
            elif 'text/markdown' in data:
                source = ''.join(data['text/markdown'])
            elif 'text/html' in data:
                source = re.sub(r'<style\b[^>]*>.*?</style>', '', ''.join(data['text/html']), flags=re.S)
                # Inline table cells wrap even when a front end strips stylesheet rules.
                source = source.replace('<table ', '<table style="width:100%;table-layout:fixed;font-size:12px" ')
                source = re.sub(r'<(td|th)(\s[^>]*)?>', lambda m: '<'+m[1]+(m[2] or '')+' style="white-space:normal;overflow-wrap:anywhere;padding:6px">', source)
            else:
                continue
            count += 1
            expanded.append({'cell_type': 'markdown', 'id': f'inline-{cell["id"]}-{index}',
                'metadata': {'code012_inline_result': True, 'source_code_cell': cell['id']},
                'source': source.splitlines(keepends=True), **({'attachments': attachments} if attachments else {})})
    notebook['cells'] = expanded
    PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n')
    print(f'Rendered {count} result blocks as full-height notebook content; execution outputs retained and hidden.')

if __name__ == '__main__':
    main()
