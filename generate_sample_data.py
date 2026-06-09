"""
Gera dados sintéticos realistas de receita e despesa diária para 10 unidades.
Execute antes do main.py para criar a base histórica de simulação.

    python generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEMENTE = 42
UNIDADES = [f"{i:03d}" for i in range(1, 11)]
DATA_INICIO = "2023-01-01"
RECEITA_BASE = 50_000
DESPESA_BASE = 40_000


def _gerar_serie_unidade(
    datas: pd.DatetimeIndex,
    base: float,
    ruido_escala: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(datas)
    dias_semana = datas.dayofweek.values
    dia_ano = datas.dayofyear.values
    dia_mes = datas.day.values

    # Fechamento dominical
    aberto = (dias_semana != 6).astype(float)

    # Sazonalidade semanal: pico na sexta, baixo na segunda
    sazon_semana = 1.0 + 0.12 * np.sin(2 * np.pi * dias_semana / 7)

    # Sazonalidade mensal: pico nos primeiros 10 dias (pagamentos)
    sazon_mes = 1.0 + 0.08 * np.exp(-0.1 * (dia_mes - 5) ** 2)

    # Sazonalidade anual: dez/jan maiores
    sazon_ano = 1.0 + 0.10 * np.cos(2 * np.pi * (dia_ano - 15) / 365)

    # Tendência de crescimento leve ao longo do período
    tendencia = 1.0 + 0.08 * np.linspace(0, 1, n)

    ruido = rng.normal(loc=1.0, scale=ruido_escala, size=n)
    ruido = np.clip(ruido, 0.6, 1.4)

    serie = base * aberto * sazon_semana * sazon_mes * sazon_ano * tendencia * ruido
    return np.round(serie, 2)


def gerar_dados(data_inicio: str = DATA_INICIO) -> pd.DataFrame:
    rng = np.random.default_rng(SEMENTE)
    hoje = pd.Timestamp.today().normalize()
    datas = pd.date_range(start=data_inicio, end=hoje - pd.Timedelta(days=1), freq="D")

    registros = []
    for unidade in UNIDADES:
        receita = _gerar_serie_unidade(datas, RECEITA_BASE, ruido_escala=0.07, rng=rng)
        despesa = _gerar_serie_unidade(datas, DESPESA_BASE, ruido_escala=0.06, rng=rng)
        df_u = pd.DataFrame({
            "ds": datas,
            "unidade": unidade,
            "receita": receita,
            "despesa": despesa,
        })
        registros.append(df_u)

    return pd.concat(registros, ignore_index=True)


def main():
    print("[INFO] Gerando dados sintéticos...")
    df = gerar_dados()

    pasta = Path("dados")
    pasta.mkdir(exist_ok=True)

    caminho_unidades = pasta / "base_diaria_unidades.csv"
    df.to_csv(caminho_unidades, index=False)
    print(f"[OK] Base por unidade salva em {caminho_unidades}  ({len(df):,} linhas)")

    df_geral = (
        df.groupby("ds", as_index=False)[["receita", "despesa"]].sum()
    )
    caminho_geral = pasta / "base_diaria.csv"
    df_geral.to_csv(caminho_geral, index=False)
    print(f"[OK] Base consolidada salva em {caminho_geral}  ({len(df_geral):,} linhas)")

    print("\nEstatísticas da base por unidade:")
    print(df.groupby("unidade")[["receita", "despesa"]].mean().round(0).to_string())


if __name__ == "__main__":
    main()
