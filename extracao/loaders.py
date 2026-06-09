"""Carregamento de dados históricos a partir de CSV ou banco de dados."""

import pandas as pd


def carregar_base_csv(caminho: str) -> pd.DataFrame:
    """Carrega base histórica de um arquivo CSV."""
    df = pd.read_csv(caminho, parse_dates=["ds"])
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def carregar_base_por_unidade(caminho: str) -> dict[str, pd.DataFrame]:
    """Carrega base histórica por unidade e retorna dicionário {unidade: DataFrame}."""
    df = carregar_base_csv(caminho)

    if "unidade" not in df.columns:
        raise ValueError("Coluna 'unidade' não encontrada no arquivo.")

    bases = {}
    for unidade, grupo in df.groupby("unidade"):
        bases[str(unidade)] = grupo.reset_index(drop=True)

    return bases


def carregar_feriados_locais(caminho: str, cidade: str | None = None) -> list[tuple[int, int]]:
    """
    Carrega feriados locais de um CSV com colunas: data, cidade (opcional), descricao.

    Formatos aceitos para a coluna 'data': AAAA-MM-DD ou MM-DD.
    Retorna lista de (mes, dia).
    """
    try:
        df = pd.read_csv(caminho)
    except FileNotFoundError:
        return []

    if "cidade" in df.columns and cidade:
        df = df[df["cidade"].isna() | (df["cidade"] == cidade)]

    feriados = []
    for data_str in df["data"].dropna().astype(str):
        partes = data_str.strip().split("-")
        if len(partes) == 3:
            feriados.append((int(partes[1]), int(partes[2])))
        elif len(partes) == 2:
            feriados.append((int(partes[0]), int(partes[1])))

    return feriados
