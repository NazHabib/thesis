import os
import io
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================
# Configuração do projecto
# =========================

@dataclass(frozen=True)
class CycleSpec:
    label: str                 # "2015-2016", "2017-2018", "2021-2023"
    public_folder: str         # "2015", "2017", "2021" (estrutura URL do CDC/NCHS)
    files: Dict[str, str]      # nome lógico -> nome XPT (sem extensão)
    bp_kind: str               # "auscultatory" ou "oscillometric"
    diet_day1_mode: str        # "MEC_in_person" ou "telephone_post_MEC"


CYCLES: List[CycleSpec] = [
    CycleSpec(
        label="2015-2016",
        public_folder="2015",
        files={
            "DEMO": "DEMO_I",
            "BP": "BPX_I",
            "BMX": "BMX_I",
            "A1C": "GHB_I",
            "TCHOL": "TCHOL_I",
            "HDL": "HDL_I",
            "SMQ": "SMQ_I",
            "PAQ": "PAQ_I",
            "DR1TOT": "DR1TOT_I",
        },
        bp_kind="auscultatory",
        diet_day1_mode="MEC_in_person",
    ),
    CycleSpec(
        label="2017-2018",
        public_folder="2017",
        files={
            "DEMO": "DEMO_J",
            "BP": "BPX_J",       
            "BMX": "BMX_J",
            "A1C": "GHB_J",
            "TCHOL": "TCHOL_J",
            "HDL": "HDL_J",
            "SMQ": "SMQ_J",
            "PAQ": "PAQ_J",
            "DR1TOT": "DR1TOT_J",
        },
        bp_kind="auscultatory",
        diet_day1_mode="MEC_in_person",
    ),
    CycleSpec(
        label="2021-2023",
        public_folder="2021",
        files={
            "DEMO": "DEMO_L",
            "BP": "BPXO_L",      # oscilométrico no ciclo 08/2021–08/2023
            "BMX": "BMX_L",
            "A1C": "GHB_L",
            "TCHOL": "TCHOL_L",
            "HDL": "HDL_L",
            "SMQ": "SMQ_L",
            "PAQ": "PAQ_L",
            "DR1TOT": "DR1TOT_L",
        },
        bp_kind="oscillometric",
        diet_day1_mode="telephone_post_MEC",
    ),
]


# =========================
# Utilitários de rede/cache
# =========================

def build_session(
    retries: int = 5,
    backoff_factor: float = 0.7,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": user_agent})
    return sess


def nhanes_xpt_url(public_folder: str, xpt_stem: str) -> str:
    return f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{public_folder}/DataFiles/{xpt_stem}.xpt"


def download_xpt(
    session: requests.Session,
    url: str,
    out_path: str,
    timeout: int = 60
) -> bool:
    """
    Faz download com cache local:
    - Se o ficheiro já existir, não volta a descarregar.
    - Retorna True se o ficheiro existir no fim, False se falhar (ex.: 404).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    r = session.get(url, timeout=timeout)
    if r.status_code == 404:
        return False
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)
    return True


def read_xpt(path: str) -> pd.DataFrame:
    # pandas lê SAS XPORT diretamente
    return pd.read_sas(path, format="xport")


# =========================
# Validações e harmonização
# =========================

def assert_one_row_per_seqn(df: pd.DataFrame, file_tag: str) -> None:
    if "SEQN" not in df.columns:
        raise ValueError(f"[{file_tag}] Falta coluna SEQN; não é possível fazer merge seguro.")
    if df["SEQN"].duplicated().any():
        # Isto acontece em ficheiros “longos” (ex.: DR1IFF) — neste projecto queremos 1 linha por pessoa.
        raise ValueError(f"[{file_tag}] SEQN não é único (ficheiro multi-linhas). Use agregação ou escolha DR1TOT.")


def safe_select(df: pd.DataFrame, cols: Optional[List[str]]) -> pd.DataFrame:
    if not cols:
        return df
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()


def harmonize_bp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmoniza colunas de pressão arterial:
    - BPX (auscultatório) usa BPXSY1..4 / BPXDI1..4 e BPXPLS (pulso manual).
    - BPXO (oscilométrico) usa BPXOSY1..3 / BPXODI1..3 e BPXOPLS1..3.
    Estratégia:
    - Normalizar nomes para BPXSY* / BPXDI* e BPXPLS* (pulso por leitura).
    - Preencher leituras inexistentes com NaN.
    """
    df = df.copy()

    # Renomear systolic/diastolic oscilométrico para o padrão "BPXSY" / "BPXDI"
    rename_map = {}
    for i in (1, 2, 3, 4):
        if f"BPXOSY{i}" in df.columns:
            rename_map[f"BPXOSY{i}"] = f"BPXSY{i}"
        if f"BPXODI{i}" in df.columns:
            rename_map[f"BPXODI{i}"] = f"BPXDI{i}"

    # Pulsos por leitura no oscilométrico
    for i in (1, 2, 3, 4):
        if f"BPXOPLS{i}" in df.columns:
            rename_map[f"BPXOPLS{i}"] = f"BPXPLS{i}"

    df.rename(columns=rename_map, inplace=True)

    # Garantir colunas standard (até 4 leituras)
    for i in (1, 2, 3, 4):
        if f"BPXSY{i}" not in df.columns:
            df[f"BPXSY{i}"] = pd.NA
        if f"BPXDI{i}" not in df.columns:
            df[f"BPXDI{i}"] = pd.NA
        if f"BPXPLS{i}" not in df.columns:
            df[f"BPXPLS{i}"] = pd.NA

    return df


def add_derived_bp_means(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sbp_cols = [f"BPXSY{i}" for i in (1, 2, 3, 4)]
    dbp_cols = [f"BPXDI{i}" for i in (1, 2, 3, 4)]
    df["SBP_mean"] = pd.to_numeric(df[sbp_cols].stack(), errors="coerce").groupby(level=0).mean()
    df["DBP_mean"] = pd.to_numeric(df[dbp_cols].stack(), errors="coerce").groupby(level=0).mean()
    return df


# =========================
# Pipeline por ciclo
# =========================

def build_cycle_dataset(
    cycle: CycleSpec,
    session: requests.Session,
    cache_dir: str = "nhanes_xpt_cache",
    polite_sleep: float = 0.2
) -> pd.DataFrame:
    """
    Constrói dataset por ciclo:
    - Base: DEMO (1 linha por pessoa).
    - Left-join dos restantes ficheiros.
    - Harmonização BP + derivados simples.
    - Adição de metadados (cycle, método BP, modo dieta).
    """
    base_url_debug = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle.public_folder}/DataFiles/"

    # 1) descarregar + ler DEMO
    demo_stem = cycle.files["DEMO"]
    demo_url = nhanes_xpt_url(cycle.public_folder, demo_stem)
    demo_path = os.path.join(cache_dir, cycle.label, f"{demo_stem}.xpt")

    ok = download_xpt(session, demo_url, demo_path)
    if not ok:
        raise FileNotFoundError(f"DEMO não encontrado: {demo_url}")

    demo = read_xpt(demo_path)
    assert_one_row_per_seqn(demo, f"{cycle.label}:{demo_stem}")

    # (opcional) manter só variáveis essenciais + pesos + desenho
    demo_keep = [
        "SEQN", "SDDSRVYR",
        "RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDRETH3",
        "DMDEDUC2", "INDHHIN2", "INDFMPIR", "DMDMARTL", "DMDMARTZ",
        "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA",
    ]
    demo = safe_select(demo, demo_keep)

    # 2) iterar restantes ficheiros e fazer left-join
    df = demo
    for logical_name, xpt_stem in cycle.files.items():
        if logical_name == "DEMO":
            continue

        url = nhanes_xpt_url(cycle.public_folder, xpt_stem)
        path = os.path.join(cache_dir, cycle.label, f"{xpt_stem}.xpt")

        ok = download_xpt(session, url, path)
        if not ok:
            # 08/2021–08/2023 tem releases parciais; é melhor continuar e registar falhas
            print(f"[AVISO] Ficheiro não encontrado (provável release parcial): {url}")
            continue

        tmp = read_xpt(path)
        assert_one_row_per_seqn(tmp, f"{cycle.label}:{xpt_stem}")

        # Seleções por ficheiro (para reduzir memória)
        if logical_name == "BP":
            tmp = harmonize_bp(tmp)
            keep = ["SEQN"] + [f"BPXSY{i}" for i in (1, 2, 3, 4)] + [f"BPXDI{i}" for i in (1, 2, 3, 4)] \
                   + ["BPXPLS"] + [f"BPXPLS{i}" for i in (1, 2, 3, 4)]
            tmp = safe_select(tmp, keep)

        elif logical_name == "BMX":
            keep = ["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST", "BMXHIP"]
            tmp = safe_select(tmp, keep)

        elif logical_name == "A1C":
            keep = ["SEQN", "LBXGH", "WTPH2YR"]  # WTPH2YR pode estar presente nos ficheiros de analitos (2021-2023)
            tmp = safe_select(tmp, keep)

        elif logical_name in ("TCHOL", "HDL"):
            # manter mg/dL (LBX*) e SI (LBD*SI) quando existir
            keep = ["SEQN", "WTPH2YR"] + [c for c in tmp.columns if c.startswith("LBX") or c.endswith("SI")]
            tmp = safe_select(tmp, keep)

        elif logical_name == "SMQ":
            keep = ["SEQN", "SMQ020", "SMQ040"]
            tmp = safe_select(tmp, keep)

        elif logical_name == "PAQ":
            keep = ["SEQN", "PAQ650", "PAQ670", "PAD680"]
            tmp = safe_select(tmp, keep)

        elif logical_name == "DR1TOT":
            # inclui pesos dietéticos (WTDRD1/WTDR2D) e macronutrientes clássicos
            keep = [
                "SEQN", "WTDRD1", "WTDR2D", "DR1DRSTZ", "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR",
                "DR1TTFAT", "DR1TSFAT", "DR1TFIBE", "DR1TSODI", "DR1TPOTA", "DR1TALCO"
            ]
            tmp = safe_select(tmp, keep)

        else:
            tmp = tmp

        df = df.merge(tmp, on="SEQN", how="left")
        time.sleep(polite_sleep)

    # 3) metadados críticos para comparabilidade
    df["Cycle"] = cycle.label
    df["BP_method"] = cycle.bp_kind
    df["Diet_day1_mode"] = cycle.diet_day1_mode

    # 4) derivados simples (úteis já na camada de ML e na safety layer)
    if "BPXSY1" in df.columns:
        df = add_derived_bp_means(df)

    return df


def compute_multi_cycle_weight(df: pd.DataFrame, weight_col: str, n_cycles: int, new_col: str) -> pd.DataFrame:
    """
    Re-scaling clássico de pesos NHANES ao combinar ciclos 2-anos.
    Não deve ser aplicado cegamente a combinações com grandes lacunas temporais; fique com isto como opção.
    """
    df = df.copy()
    if weight_col in df.columns:
        df[new_col] = pd.to_numeric(df[weight_col], errors="coerce") / float(n_cycles)
    return df


# =========================
# Execução total
# =========================

def main(out_dir: str = "nhanes_out") -> None:
    os.makedirs(out_dir, exist_ok=True)
    session = build_session()

    cycle_frames = []
    manifest = {"generated_at": pd.Timestamp.utcnow().isoformat(), "cycles": []}

    for cycle in CYCLES:
        print(f"\n=== A processar ciclo {cycle.label} ===")
        df_cycle = build_cycle_dataset(cycle, session=session)

        # Guardar por ciclo (melhor para debug e auditoria)
        cycle_parquet = os.path.join(out_dir, f"NHANES_{cycle.label}_raw.parquet")
        cycle_csv = os.path.join(out_dir, f"NHANES_{cycle.label}_raw.csv")

        df_cycle.to_parquet(cycle_parquet, index=False)
        df_cycle.to_csv(cycle_csv, index=False)

        manifest["cycles"].append({
            **asdict(cycle),
            "rows": int(df_cycle.shape[0]),
            "cols": int(df_cycle.shape[1]),
            "outputs": [cycle_parquet, cycle_csv],
        })

        cycle_frames.append(df_cycle)

    master = pd.concat(cycle_frames, ignore_index=True, sort=False)

    # Exemplo opcional: pesos multi-ciclo (apenas como “utility”; use com critério)
    master = compute_multi_cycle_weight(master, "WTMEC2YR", n_cycles=len(CYCLES), new_col="WTMEC_multi_cycle")
    master = compute_multi_cycle_weight(master, "WTDRD1", n_cycles=len(CYCLES), new_col="WTDRD1_multi_cycle")
    master = compute_multi_cycle_weight(master, "WTPH2YR", n_cycles=len(CYCLES), new_col="WTPH_multi_cycle")

    master_parquet = os.path.join(out_dir, "NHANES_Master_raw.parquet")
    master_csv = os.path.join(out_dir, "NHANES_Master_raw.csv")
    master.to_parquet(master_parquet, index=False)
    master.to_csv(master_csv, index=False)

    manifest["master"] = {"rows": int(master.shape[0]), "cols": int(master.shape[1]),
                          "outputs": [master_parquet, master_csv]}

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nOK. Master dataset: {master.shape[0]} linhas x {master.shape[1]} colunas.")
    print(f"Saídas em: {out_dir}/")


if __name__ == "__main__":
    main()
