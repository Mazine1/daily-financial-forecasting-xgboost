"""Geração de features de calendário e engenharia de features para o modelo."""

import pandas as pd


# ---------------------------------------------------------------------------
# Feriados nacionais brasileiros (algoritmo de Páscoa de Butcher)
# ---------------------------------------------------------------------------

def calcular_pascoa(ano: int) -> pd.Timestamp:
    """Calcula a data da Páscoa usando o algoritmo de Butcher."""
    a = ano % 19
    b = ano // 100
    c = ano % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(ano, mes, dia)


def feriados_nacionais(ano: int) -> list[pd.Timestamp]:
    """Retorna lista de feriados nacionais brasileiros para o ano."""
    pascoa = calcular_pascoa(ano)
    feriados = [
        pd.Timestamp(ano, 1, 1),   # Confraternização Universal
        pd.Timestamp(ano, 4, 21),  # Tiradentes
        pd.Timestamp(ano, 5, 1),   # Dia do Trabalho
        pd.Timestamp(ano, 9, 7),   # Independência
        pd.Timestamp(ano, 10, 12), # N. Sra. Aparecida
        pd.Timestamp(ano, 11, 2),  # Finados
        pd.Timestamp(ano, 11, 15), # Proclamação da República
        pd.Timestamp(ano, 12, 25), # Natal
        pascoa - pd.Timedelta(days=2),   # Sexta-feira Santa
        pascoa,                          # Páscoa
        pascoa + pd.Timedelta(days=60),  # Corpus Christi
    ]
    return sorted(feriados)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _flag_janela(datas: pd.Series, eventos: list[pd.Timestamp], dias: int, direcao: str) -> pd.Series:
    """Gera flag 0/1 para janela de 'dias' antes ('antes') ou depois ('depois') de eventos."""
    flag = pd.Series(0, index=datas.index)
    for evento in eventos:
        if direcao == "antes":
            mascara = (datas >= evento - pd.Timedelta(days=dias)) & (datas < evento)
        else:
            mascara = (datas > evento) & (datas <= evento + pd.Timedelta(days=dias))
        flag[mascara] = 1
    return flag


def adicionar_features_calendario(
    df: pd.DataFrame,
    col_data: str = "ds",
    feriados_locais: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """
    Adiciona features de calendário, feriados nacionais e locais ao DataFrame.

    Parâmetros
    ----------
    df : DataFrame com coluna de data
    col_data : nome da coluna de data
    feriados_locais : lista de (mes, dia) de feriados locais fixos
    """
    df = df.copy()
    datas = pd.to_datetime(df[col_data])

    df["dia"] = datas.dt.day
    df["mes"] = datas.dt.month
    df["ano"] = datas.dt.year
    df["dia_semana"] = datas.dt.dayofweek          # 0=seg, 6=dom
    df["dia_ano"] = datas.dt.dayofyear
    df["semana_mes"] = (datas.dt.day - 1) // 7 + 1
    df["fim_mes"] = (datas + pd.offsets.MonthEnd(0) == datas).astype(int)
    df["inicio_mes"] = (datas.dt.day == 1).astype(int)
    df["dias_para_fim_mes"] = (datas + pd.offsets.MonthEnd(0) - datas).dt.days
    df["janela_inicio_mes_5d"] = (datas.dt.day <= 5).astype(int)
    df["janela_fim_mes_5d"] = (df["dias_para_fim_mes"] <= 5).astype(int)

    # Feriados nacionais por ano presente nos dados
    anos = datas.dt.year.unique()
    todos_feriados_nac = []
    for ano in anos:
        todos_feriados_nac.extend(feriados_nacionais(ano))

    conjunto_feriados_nac = set(f.normalize() for f in todos_feriados_nac)
    df["feriado_nacional"] = datas.dt.normalize().isin(conjunto_feriados_nac).astype(int)
    df["prox_feriado_7d"] = _flag_janela(datas, todos_feriados_nac, 7, "antes")
    df["pos_feriado_7d"] = _flag_janela(datas, todos_feriados_nac, 7, "depois")

    # Feriados locais
    if feriados_locais:
        datas_feriados_loc = []
        for ano in anos:
            for mes, dia in feriados_locais:
                try:
                    datas_feriados_loc.append(pd.Timestamp(ano, mes, dia))
                except ValueError:
                    pass

        conjunto_feriados_loc = set(f.normalize() for f in datas_feriados_loc)
        df["feriado_local"] = datas.dt.normalize().isin(conjunto_feriados_loc).astype(int)
        df["prox_feriado_local_7d"] = _flag_janela(datas, datas_feriados_loc, 7, "antes")
        df["pos_feriado_local_7d"] = _flag_janela(datas, datas_feriados_loc, 7, "depois")
    else:
        df["feriado_local"] = 0
        df["prox_feriado_local_7d"] = 0
        df["pos_feriado_local_7d"] = 0

    return df


COLUNAS_CALENDARIO = [
    "dia", "mes", "ano", "dia_semana", "dia_ano", "semana_mes",
    "fim_mes", "inicio_mes", "dias_para_fim_mes",
    "janela_inicio_mes_5d", "janela_fim_mes_5d",
    "feriado_nacional", "prox_feriado_7d", "pos_feriado_7d",
    "feriado_local", "prox_feriado_local_7d", "pos_feriado_local_7d",
]


def adicionar_lags_e_rolling(
    df: pd.DataFrame,
    col_alvo: str,
    lags: list[int],
    rolls: list[int],
    growth_windows: list[int],
) -> pd.DataFrame:
    """Adiciona lags, médias móveis e features de crescimento/tendência."""
    df = df.copy().sort_values("ds").reset_index(drop=True)

    for lag in lags:
        df[f"lag_{lag}"] = df[col_alvo].shift(lag)

    for janela in rolls:
        df[f"roll_mean_{janela}"] = df[col_alvo].shift(1).rolling(janela).mean()

    for janela in growth_windows:
        roll_atual = df[col_alvo].shift(1).rolling(janela).mean()
        roll_anterior = df[col_alvo].shift(janela + 1).rolling(janela).mean()
        df[f"crescimento_{janela}d"] = (roll_atual - roll_anterior) / (roll_anterior.abs() + 1e-6)
        df[f"tendencia_{janela}d"] = roll_atual - roll_anterior

    return df
