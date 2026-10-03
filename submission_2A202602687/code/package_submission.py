"""Package the deliverable without datasets, checkpoints or Python/Jupyter caches."""
from pathlib import Path
import zipfile

def package(out):
    out=Path(out).resolve(); target=out.with_suffix(".zip")
    excluded={"__pycache__",".ipynb_checkpoints",".DS_Store"}
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in sorted(out.rglob("*")):
            if not file.is_file(): continue
            relative=file.relative_to(out)
            if any(part in excluded for part in relative.parts): continue
            if file.suffix in (".pyc",".pt",".pth",".ckpt",".npz"): continue
            archive.write(file,Path(out.name)/relative)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        names=archive.namelist()
        for name in ("REPORT.md","experiments.xlsx","predictions_eval.csv","eval_result.json","code/lab.ipynb"):
            assert f"{out.name}/{name}" in names
        assert not any("__pycache__" in name or name.endswith((".pt",".npz")) for name in names)
    print(f"Created {target}: {target.stat().st_size/2**20:.2f} MB, {len(names)} files")
    return target

if __name__=="__main__":
    import sys
    package(sys.argv[1])
