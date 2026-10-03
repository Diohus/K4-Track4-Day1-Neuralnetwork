"""Persist measured histories and fill the unchanged four-sheet Excel template."""
import copy
import json
import math
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import openpyxl
from openpyxl.formula.translate import Translator
from openpyxl.workbook.properties import CalcProperties

FORMULA_COLUMNS = {"step0_gap_vs_lnC","gap_val_minus_train","delta_val_f1_vs_base","beyond_noise"}

def save_result(result, results_dir="../results"):
    out=Path(results_dir)/f"{result['cfg']['exp_id']}.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    payload={k:result[k] for k in ("cfg","history","summary")}
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return str(out)

def load_results(results_dir="../results"):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(results_dir).glob("*.json"))
            if p.name != "health.json"]

def to_row(result, eval_scores=None, notes=""):
    cfg=result["cfg"]
    durations=result["history"].get("epoch_time_s",[])
    timing=""
    if durations and max(durations)>10*np.median(durations):
        timing=f"Timing outlier: max epoch {max(durations):.3f}s, median {np.median(durations):.3f}s; arithmetic mean retained, do not infer optimizer speed from this run."
    row={**cfg,**result["summary"],"figure_file":f"figures/{cfg['exp_id']}.png",
         "notes":"; ".join(filter(None,[cfg.get("notes",""),notes,result["summary"].get("failure_reason",""),timing,
                   f"activation_std={result['summary'].get('activation_std')}; mean clip_fraction={np.mean(result['history']['clip_fraction']) if result['history'].get('clip_fraction') else None}"]))}
    row["hidden"]="-".join(map(str,cfg["hidden"]))
    row["clip_norm"]="none" if cfg["clip_norm"] is None else cfg["clip_norm"]
    if eval_scores:
        row.update(eval_acc=eval_scores["accuracy"],eval_macro_f1=eval_scores["macro_f1"])
    return row

def write_xlsx(rows,template_path,out_path):
    wb=openpyxl.load_workbook(template_path)
    ws=wb["Experiments"]
    headers=[c.value for c in ws[1]]
    last=max(61,len(rows)+1)
    # Clear the prefilled example and any old input cells; preserve formula columns.
    for line in ws.iter_rows(min_row=2,max_row=last):
        for cell in line:
            if headers[cell.column-1] not in FORMULA_COLUMNS: cell.value=None
    for i,row in enumerate(rows,2):
        for j,key in enumerate(headers,1):
            cell=ws.cell(i,j)
            if key in FORMULA_COLUMNS:
                cell.value=Translator(ws.cell(2,j).value,origin=ws.cell(2,j).coordinate).translate_formula(cell.coordinate)
            else:
                cell.value=row.get(key)
            if i>61:
                cell._style=copy.copy(ws.cell(2,j)._style)
    baseline=[r for r in rows if r["group"]=="baseline"]
    seeds=wb["Seeds"]
    if len(baseline)>5: raise ValueError("Template supports at most five baseline seeds")
    for i in range(2,7): seeds.cell(i,1).value=baseline[i-2]["exp_id"] if i-2<len(baseline) else None
    # Expand lookup ranges if more than the template's 60 runs are used.
    if last>61:
        for sheet in wb:
            for line in sheet:
                for cell in line:
                    if cell.data_type=="f": cell.value=cell.value.replace("$61",f"${last}")
    for i in range(2,wb["Summary"].max_row+1):
        group=wb["Summary"].cell(i,1).value
        subset=[r for r in rows if r["group"]==group and not r.get("diverged")]
        if subset:
            best=max(subset,key=lambda r:r.get("val_macro_f1") or 0)
            wb["Summary"].cell(i,8).value=f"Best measured validation F1: {best['exp_id']} ({best['val_macro_f1']:.4f}); consult report for seed limitations."
    wb.calculation=CalcProperties(calcId=191029,fullCalcOnLoad=True)
    Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    wb.save(out_path)
    _cache_formula_values(out_path,rows)


def _cache_formula_values(path,rows):
    """Store formula caches so headless readers see results; retain original formulas.

    openpyxl does not calculate Excel formulas. Values below evaluate this template's
    four computed columns and Seeds/Summary tables from the same measured rows.
    Excel/LibreOffice can still recalculate all formulas on opening.
    """
    ns={"s":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    tag="{"+ns["s"]+"}"
    by_id={r["exp_id"]:r for r in rows}
    base=[r for r in rows if r["group"]=="baseline"]
    mean=lambda k:float(np.mean([r[k] for r in base])) if base else None
    std=lambda k:float(np.std([r[k] for r in base],ddof=1)) if len(base)>1 else None
    sigma=std("val_macro_f1"); baseline_mean=mean("val_macro_f1")
    cache={2:{},3:{},4:{}}
    for i,r in enumerate(rows,2):
        cache[2][f"AD{i}"]=r["step0_loss"]-math.log(7) if r.get("step0_loss") is not None else None
        cache[2][f"AE{i}"]=r["final_val_loss"]-r["final_train_loss"] if r.get("final_val_loss") is not None else None
        delta=r["val_macro_f1"]-baseline_mean if baseline_mean is not None else None
        cache[2][f"AF{i}"]=delta
        cache[2][f"AG{i}"]=("C?" if abs(delta)>2*sigma else "Kh?ng") if sigma is not None and delta is not None else None
    for i,r in enumerate(base,2):
        for col,key in (("B","val_acc"),("C","val_macro_f1"),("D","best_val_loss")):
            cache[3][f"{col}{i}"]=r.get(key)
    for col,key in (("B","val_acc"),("C","val_macro_f1"),("D","best_val_loss")):
        cache[3][f"{col}8"]=mean(key)
        cache[3][f"{col}9"]=std(key)
        cache[3][f"{col}10"]=2*std(key) if std(key) is not None else None
    with zipfile.ZipFile(path) as archive:
        contents={n:archive.read(n) for n in archive.namelist()}
    summary=ET.fromstring(contents["xl/worksheets/sheet4.xml"])
    for line in summary.findall("s:sheetData/s:row",ns):
        i=int(line.get("r")); group=None
        for cell in line:
            if cell.get("r")==f"A{i}":
                inline=cell.find("s:is",ns)
                if inline is not None: group="".join(inline.itertext())
        if group is None or i==1: continue
        subset=[r for r in rows if r["group"]==group]
        valid=[r for r in subset if r.get("val_macro_f1") is not None]
        cache[4].update({f"B{i}":len(subset),f"C{i}":len(valid),
            f"D{i}":max((r["val_macro_f1"] for r in valid),default=None),
            f"E{i}":min((r["val_macro_f1"] for r in valid),default=None),
            f"F{i}":max((r["val_acc"] for r in valid),default=None),
            f"G{i}":"C?" if valid else "Ch?a"})
    for sheet in (2,3,4):
        name=f"xl/worksheets/sheet{sheet}.xml"
        root=ET.fromstring(contents[name])
        for c in root.findall(".//s:c",ns):
            if c.find("s:f",ns) is None: continue
            val=cache[sheet].get(c.get("r"))
            old=c.find("s:v",ns)
            if old is not None: c.remove(old)
            v=ET.SubElement(c,tag+"v")
            if val is None:
                c.set("t","str"); v.text=""
            elif isinstance(val,str):
                c.set("t","str"); v.text=val
            else:
                c.attrib.pop("t",None)
                v.text=str(val)
        contents[name]=ET.tostring(root,encoding="utf-8",xml_declaration=True)
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,content in contents.items(): archive.writestr(name,content)
