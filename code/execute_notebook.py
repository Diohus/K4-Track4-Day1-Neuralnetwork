"""Execute every cell in a fresh kernel and preserve real outputs in the notebook."""
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

def execute(path):
    path=Path(path).resolve()
    notebook=nbformat.read(path,as_version=4)
    manager=KernelManager(kernel_name="python3")
    manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    client=NotebookClient(notebook,km=manager,timeout=3600,
        resources={"metadata":{"path":str(path.parent)}},allow_errors=False)
    try:
        client.execute()
    finally:
        # Keep partial outputs on failure to make debugging reviewable.
        nbformat.write(notebook,path)
    errors=[o for c in notebook.cells if c.cell_type=="code" for o in c.get("outputs",[]) if o.output_type=="error"]
    assert not errors
    print(f"Fresh-kernel Run All passed: {sum(c.cell_type=='code' for c in notebook.cells)} code cells, real outputs saved to {path}")

if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"]="utf-8"
    execute(sys.argv[1])
