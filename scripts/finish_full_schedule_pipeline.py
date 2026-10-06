"""Wait for the active Code 08 kernel, then execute Codes 09/10 and save exports."""
from pathlib import Path
import json,sys,time,traceback
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/paper_matched_full_schedule'
STATUS=OUT/'pipeline_status.json'
LOG=OUT/'pipeline_execution.log'
NOTEBOOKS=['08_full_schedule_comparison','09_full_schedule_oos','10_regime_analysis']

def status(stage,**extra):
    payload={'stage':stage,'updated_unix':time.time(),**extra}
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2));temp.replace(STATUS)

class LiveClient(NotebookClient):
    def process_message(self,msg,cell,cell_index):
        if msg['msg_type']=='stream':
            with LOG.open('a') as f:f.write(msg['content']['text'])
        return super().process_message(msg,cell,cell_index)

def execute(name):
    path=ROOT/'notebooks'/f'{name}.ipynb'
    n=nbformat.read(path,as_version=4)
    client=LiveClient(n,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.create_kernel_manager()
    client.km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:client.execute()
    finally:nbformat.write(n,path)

def export(name,answer_path):
    path=ROOT/'notebooks'/f'{name}.ipynb';n=nbformat.read(path,as_version=4)
    n.cells=[c for c in n.cells if c.get('id')!='executed-conclusions']
    summary=answer_path.read_text()
    # The executed output contains formatted tables; preserve its text as a source section too.
    n.cells.append(nbformat.v4.new_markdown_cell('## Saved run conclusions\n\n'+summary))
    n.cells[-1].id='executed-conclusions'
    nbformat.write(n,path)
    parts=[]
    for c in n.cells:
        s=c.source
        parts.append('# %% [markdown]\n'+'\n'.join('# '+l for l in s.splitlines()) if c.cell_type=='markdown' else '# %%\n'+s)
    path.with_suffix('.py').write_text('\n\n'.join(parts)+'\n')
    errors=[o for c in n.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']
    assert not errors
    assert all(c.execution_count is not None for c in n.cells if c.cell_type=='code' and c.source.strip())

try:
    status('waiting_for_code08',message='Author-convention 18-model training is running in its existing kernel.')
    while True:
        try:n=nbformat.read(ROOT/'notebooks/08_full_schedule_comparison.ipynb',as_version=4)
        except (OSError,ValueError):time.sleep(30);continue
        errors=[o for c in n.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']
        if errors:raise RuntimeError('Code 08 ended with error: '+str(errors[-1].get('evalue')))
        codes=[c for c in n.cells if c.cell_type=='code' and c.source.strip()]
        if all(c.execution_count is not None for c in codes) and (OUT/'tables/full_short_paper_sharpe_comparison.csv').exists():break
        time.sleep(30)
    assert len(list((OUT/'models').glob('*.pt')))==18
    export(NOTEBOOKS[0],OUT/'results_and_answers.txt')
    status('running_code09')
    execute(NOTEBOOKS[1])
    export(NOTEBOOKS[1],ROOT/'results/rolling_oos_full_schedule/results_and_answers.txt')
    status('running_code10')
    execute(NOTEBOOKS[2])
    export(NOTEBOOKS[2],ROOT/'results/regime_analysis/results_and_answers.txt')
    status('complete',notebooks=NOTEBOOKS,message='All three notebooks executed without errors; exports, tables, figures, and answers saved.')
    print('FULL-SCHEDULE PIPELINE COMPLETE',flush=True)
except BaseException as exc:
    status('failed',error=str(exc),traceback=traceback.format_exc())
    with LOG.open('a') as f:f.write(traceback.format_exc())
    raise
