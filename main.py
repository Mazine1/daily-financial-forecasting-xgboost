"""
Previsão Financeira Diária com XGBoost
========================================
Executa o pipeline completo:
  1. Carrega base histórica (gerada por generate_sample_data.py)
  2. Treina um modelo XGBoost de receita e despesa por unidade
  3. Gera previsão iterativa de 60 dias à frente
  4. Exibe tabela formatada e salva localmente
  5. (Opcional) Envia ao banco de dados

Pré-requisito: rode `python generate_sample_data.py` antes da primeira execução.
"""

import os
import warnings
from pathlib import Path

import pandas as pd

from previsoes.previsao_unificada import gerar_previsoes, salvar_previsao_local
from treinadores.treinar_xgboost import treinar_todos_por_unidade

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
UNIDADES = [f"{i:03d}" for i in range(1, 11)]
CAMINHO_BASE_UNIDADES = "dados/base_diaria_unidades.csv"
DIAS_PREVISAO = 60
ENVIAR_DB = False  # Altere para True após configurar o .env com as credenciais do banco


# ---------------------------------------------------------------------------
# Utilitários de exibição
# ---------------------------------------------------------------------------

def _formatar_brl(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def imprimir_tabela_previsao(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"  PREVISÃO — PRÓXIMOS {DIAS_PREVISAO} DIAS")
    print("=" * 60 + "\n")

    df_fmt = df[df["unidade"] == "TOTAL"].copy()
    df_fmt["ds"] = pd.to_datetime(df_fmt["ds"]).dt.strftime("%d/%m/%Y")
    df_fmt["receita_prev"] = df_fmt["receita_prev"].map(_formatar_brl)
    df_fmt["despesa_prev"] = df_fmt["despesa_prev"].map(_formatar_brl)
    df_fmt["saldo_prev"] = df_fmt["saldo_prev"].map(_formatar_brl)

    print(
        df_fmt[["ds", "receita_prev", "despesa_prev", "saldo_prev"]]
        .rename(columns={
            "ds": "Data",
            "receita_prev": "Receita Prev.",
            "despesa_prev": "Despesa Prev.",
            "saldo_prev": "Saldo Prev.",
        })
        .to_string(index=False)
    )


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    os.chdir(BASE_DIR)

    if not Path(CAMINHO_BASE_UNIDADES).exists():
        raise FileNotFoundError(
            f"Base histórica não encontrada em '{CAMINHO_BASE_UNIDADES}'.\n"
            "Execute primeiro:  python generate_sample_data.py"
        )

    print("\n==============================")
    print("  CARREGANDO BASE HISTÓRICA   ")
    print("==============================\n")

    df = pd.read_csv(CAMINHO_BASE_UNIDADES, parse_dates=["ds"])
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    print(f"[OK] {len(df):,} registros carregados | período: {df['ds'].min().date()} → {df['ds'].max().date()}")

    print("\n==============================")
    print("  TREINAMENTO XGBOOST         ")
    print("==============================\n")

    treinar_todos_por_unidade(df, UNIDADES)

    print("\n==============================")
    print("  GERANDO PREVISÕES           ")
    print("==============================\n")

    previsao = gerar_previsoes(
        dias=DIAS_PREVISAO,
        unidades=UNIDADES,
        caminho_base=CAMINHO_BASE_UNIDADES,
    )
    salvar_previsao_local(previsao)

    if ENVIAR_DB:
        from previsoes.salvar_previsao_db import salvar_previsao_db
        salvar_previsao_db(previsao)
    else:
        print("[INFO] Envio ao banco desativado (ENVIAR_DB = False).")

    imprimir_tabela_previsao(previsao)

    print("\n==============================")
    print("  PROCESSO CONCLUÍDO!         ")
    print("==============================\n")
