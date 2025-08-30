
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import pandas as pd
import yaml
from datetime import datetime

class SignalValidationError(Exception):
    pass

@dataclass
class StrategySpec:
    id: str
    path: Path
    format: str = "csv"
    column_mapping: Dict[str, str] = None  # e.g., {"trade_date":"trade_date","gvkey":"asset_id","weights":"target_weight"}

REQUIRED_COLS = {"trade_date", "asset_id", "target_weight"}

def load_manifest(manifest_path: Path) -> List[StrategySpec]:
    conf = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    specs: List[StrategySpec] = []
    for s in conf.get("strategies", []):
        specs.append(StrategySpec(
            id=s["id"],
            path=Path(s["path"]),
            format=s.get("format","csv"),
            column_mapping=s.get("column_mapping", {}) or {}
        ))
    return specs

def _read_file(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(path)
    elif fmt in ("xlsx","xls"):
        return pd.read_excel(path)
    elif fmt == "parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

def _standardize_columns(df: pd.DataFrame, mapping: Dict[str,str]) -> pd.DataFrame:
    df = df.rename(columns=mapping).copy()
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SignalValidationError(f"Missing required columns: {missing}")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["asset_id"] = df["asset_id"]
    df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce")
    return df

def load_strategy_file(spec: StrategySpec) -> pd.DataFrame:
    if not spec.path.exists():
        return pd.DataFrame(columns=list(REQUIRED_COLS | {"strategy_id","asof"}))
    df = _read_file(spec.path, spec.format)
    df = _standardize_columns(df, spec.column_mapping)
    df["strategy_id"] = spec.id
    if "asof" in df.columns:
        df["asof"] = pd.to_datetime(df["asof"])
    else:
        df["asof"] = datetime.fromtimestamp(spec.path.stat().st_mtime)
    return df

def load_all_strategies(specs: Sequence[StrategySpec]) -> pd.DataFrame:
    frames = [load_strategy_file(s) for s in specs]
    if len(frames)==0 or all(len(f)==0 for f in frames):
        return pd.DataFrame(columns=list(REQUIRED_COLS | {"strategy_id","asof"}))
    df = pd.concat(frames, ignore_index=True, sort=False)
    keep = ["trade_date","asset_id","target_weight","strategy_id","asof"]
    keep += [c for c in df.columns if c not in keep]
    return df[keep]

def pick_latest_per_strategy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df.copy()
    df = df.sort_values("asof").copy()
    last = df.groupby(["strategy_id","trade_date","asset_id"], as_index=False).tail(1)
    return last.reset_index(drop=True)

def merge_for_date(df: pd.DataFrame, trade_date, mode: str="select", blend: Optional[Dict[str,float]]=None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    d = df[df["trade_date"]==pd.to_datetime(trade_date).date()].copy()
    if d.empty:
        return pd.DataFrame(columns=["trade_date","asset_id","target_weight"])
    if blend is not None:
        d = d[d["strategy_id"].isin(blend.keys())]
        if d.empty:
            return pd.DataFrame(columns=["trade_date","asset_id","target_weight"])
    if mode=="select":
        if blend is None or len(blend)==0:
            pick = d["strategy_id"].value_counts().idxmax()
        else:
            pick = max(blend.items(), key=lambda kv: kv[1])[0]
        out = d[d["strategy_id"]==pick].copy()
        out = out[["trade_date","asset_id","target_weight"]]
    elif mode=="blend":
        if blend is None or sum(blend.values())<=0:
            raise SignalValidationError("blend mode requires positive weights")
        d["__w"] = d["strategy_id"].map(blend).fillna(0.0)
        d["__tw"] = d["target_weight"] * d["__w"]
        out = d.groupby(["trade_date","asset_id"], as_index=False)["__tw"].sum().rename(columns={"__tw":"target_weight"})
    else:
        raise ValueError("mode must be 'select' or 'blend'")
    return out.sort_values(["asset_id"]).reset_index(drop=True)

def enforce_constraints_and_normalize(df: pd.DataFrame, allow_short: bool=False, allow_cash: bool=False) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if not allow_short and (df["target_weight"] < 0).any():
        neg_rows = df[df["target_weight"]<0]
        raise SignalValidationError(f"Negative weights not allowed; found {len(neg_rows)} rows")
    s = df["target_weight"].sum()
    if not allow_cash:
        if s <= 0:
            raise SignalValidationError("Sum of weights <= 0; cannot normalize to 1 without allowing cash")
        df = df.copy()
        df["target_weight"] = df["target_weight"] / s
        s2 = df["target_weight"].sum()
        if abs(s2 - 1.0) > 1e-6:
            df["target_weight"] = df["target_weight"] / s2
    return df.sort_values(["asset_id"]).reset_index(drop=True)

def save_targets(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    elif out_path.suffix.lower() in (".parquet",".pq"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path.with_suffix(".csv"), index=False)
