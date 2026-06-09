"""Geração de previsão unificada de receita e despesa por unidade."""

import os

import pandas as pd

from previsoes.prever_xgboost import prever


def _listar_unidades(caminho_base: str) -> list[str]:
    df = pd.read_csv(caminho_base, usecols=["unidade"])
    if df.empty:
        return []
    s = df["unidade"].astype(str)
    pad = int(s.str.len().max()) if not s.empty else 1
    return sorted(s.str.zfill(pad).unique().tolist())


def gerar_previsoes(
    dias: int = 60,
    unidades: list[str] | None = None,
    caminho_base: str = "dados/base_diaria_unidades.csv",
) -> pd.DataFrame:
    """
    Gera previsão de receita e despesa para cada unidade e consolida o total.

    Retorna DataFrame com colunas: ds, unidade, receita_prev, despesa_prev, saldo_prev
    (mais linha 'TOTAL' por data).
    """
    print("\n==============================")
    print("  PREVISÃO XGBOOST UNIFICADA  ")
    print("==============================")

    if unidades is None:
        unidades = _listar_unidades(caminho_base)

    previsoes = []
    for unidade in unidades:
        caminho_rec = f"modelos/xgb_receita_{unidade}.pkl"
        caminho_desp = f"modelos/xgb_despesa_{unidade}.pkl"
        if not os.path.exists(caminho_rec) or not os.path.exists(caminho_desp):
            print(f"[AVISO] Modelo ausente para unidade {unidade}. Pulando.")
            continue

        try:
            prev_rec = prever(caminho_rec, "receita", dias, unidade=unidade, caminho_base=caminho_base)
            prev_desp = prever(caminho_desp, "despesa", dias, unidade=unidade, caminho_base=caminho_base)
        except Exception as exc:
            print(f"[AVISO] Erro ao prever unidade {unidade}: {exc}. Pulando.")
            continue

        df_u = prev_rec.rename(columns={"yhat": "receita_prev"}).merge(
            prev_desp.rename(columns={"yhat": "despesa_prev"}), on="ds", how="left"
        )
        df_u["unidade"] = unidade
        df_u["saldo_prev"] = df_u["receita_prev"] - df_u["despesa_prev"]
        previsoes.append(df_u)

    if not previsoes:
        raise ValueError("Nenhuma previsão gerada. Verifique os modelos e a base por unidade.")

    df_unidades = pd.concat(previsoes, ignore_index=True)

    total = (
        df_unidades.groupby("ds", as_index=False)[["receita_prev", "despesa_prev"]]
        .sum()
    )
    total["unidade"] = "TOTAL"
    total["saldo_prev"] = total["receita_prev"] - total["despesa_prev"]

    return pd.concat([df_unidades, total], ignore_index=True).sort_values(["ds", "unidade"])


def salvar_previsao_local(df: pd.DataFrame, caminho: str = "previsoes/historico_previsoes.csv") -> None:
    """Salva previsões em CSV local com carimbo de data/hora de geração."""
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    registro = df.copy()
    registro["gerado_em"] = pd.Timestamp.now()
    header = not os.path.exists(caminho)
    registro.to_csv(caminho, mode="a", header=header, index=False)
    print(f"[OK] Previsão salva localmente em {caminho}")
