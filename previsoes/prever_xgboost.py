"""Previsão iterativa de 60 dias usando modelos XGBoost treinados."""

import pickle
from typing import Dict, List, Tuple

import pandas as pd

from treinadores.treinar_xgboost import add_calendar_features, add_target_lags

PREDICTION_CAP_MULT = 1.15
FERIADOS_LOCAIS_PATH = "dados/feriados_locais.csv"


def _aplicar_caps(serie: pd.Series, caps: Tuple[float, float] | None) -> pd.Series:
    if not caps:
        return serie
    low, high = caps
    return pd.to_numeric(serie, errors="coerce").clip(lower=low, upper=high)


def _caps_com_margem(caps: Tuple[float, float] | None) -> Tuple[float, float] | None:
    if not caps:
        return None
    low, high = caps
    return (low, high * PREDICTION_CAP_MULT)


def _gerar_linha_features(
    base: pd.DataFrame,
    alvo: str,
    lags: List[int],
    rolls: List[int],
    growth_windows: List[int],
    features: List[str],
    caps: Tuple[float, float] | None,
) -> Tuple[pd.Timestamp, pd.DataFrame]:
    prox_ds = base["ds"].max() + pd.Timedelta(days=1)
    base_fut = base.copy()
    base_fut.loc[len(base_fut)] = {"ds": prox_ds, alvo: float("nan")}

    df_feat = add_calendar_features(base_fut)
    df_feat = add_target_lags(df_feat, alvo, lags, rolls, growth_windows)

    X = df_feat.iloc[-1][features].to_frame().T.apply(pd.to_numeric, errors="coerce")
    colunas_alvo = [c for c in features if c.startswith(f"{alvo}_")]
    for col in colunas_alvo:
        X[col] = _aplicar_caps(X[col], caps)
    return prox_ds, X


def _carregar_historico(
    caminho_base: str,
    alvo: str,
    unidade: str | None,
    caps: Tuple[float, float] | None,
) -> pd.DataFrame:
    df = pd.read_csv(caminho_base)
    df.columns = df.columns.str.strip()
    col_data = next((c for c in ["ds", "data"] if c in df.columns), None)
    if col_data is None:
        raise ValueError(f"Coluna de data não encontrada em {caminho_base}.")
    df = df.rename(columns={col_data: "ds"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    if unidade is not None:
        if "unidade" not in df.columns:
            raise ValueError("Coluna 'unidade' não encontrada na base.")
        u = str(unidade)
        df["unidade"] = df["unidade"].astype(str).str.zfill(len(u))
        df = df[df["unidade"] == u].copy()
        if df.empty:
            raise ValueError(f"Histórico vazio para unidade {u}.")

    df = df.sort_values("ds")[["ds", alvo]].copy()
    df[alvo] = _aplicar_caps(pd.to_numeric(df[alvo], errors="coerce"), caps)
    return df


def prever(
    caminho_modelo: str,
    alvo: str,
    dias: int = 60,
    unidade: str | None = None,
    caminho_base: str = "dados/base_diaria_unidades.csv",
) -> pd.DataFrame:
    """Previsão iterativa: cada dia previsto alimenta o próximo."""
    with open(caminho_modelo, "rb") as f:
        salvo: Dict = pickle.load(f)

    model = salvo["model"]
    features = salvo["features"]
    lags = salvo.get("lags", [])
    rolls = salvo.get("rolls", [])
    gw = salvo.get("growth_windows", [])
    caps = salvo.get("outlier_caps")
    caps_pred = _caps_com_margem(caps)

    hist = _carregar_historico(caminho_base, alvo, unidade, caps)
    janela_min = max(lags + rolls + gw) if (lags or rolls or gw) else 1
    if len(hist) < janela_min:
        raise ValueError(f"Histórico insuficiente para gerar features de '{alvo}'.")

    previsoes = []
    base = hist.reset_index(drop=True)

    for _ in range(dias):
        prox_ds, X = _gerar_linha_features(base, alvo, lags, rolls, gw, features, caps)
        yhat = float(model.predict(X)[0])
        if caps_pred:
            yhat = float(_aplicar_caps(pd.Series([yhat]), caps_pred).iloc[0])
        previsoes.append({"ds": prox_ds, "yhat": yhat})
        nova_linha = pd.DataFrame({"ds": [prox_ds], alvo: [yhat]})
        base = pd.concat([base, nova_linha], ignore_index=True)

    return pd.DataFrame(previsoes, columns=["ds", "yhat"])
