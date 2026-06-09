"""
Input validation module for the financial forecasting pipeline.

Catches malformed, manipulated, or corrupted data before it reaches the model,
preventing silent errors, data leakage, and garbage-in/garbage-out predictions.
"""

from dataclasses import dataclass, field
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLUNAS_OBRIGATORIAS = {"ds", "receita", "despesa"}
COLUNAS_UNIDADE = {"ds", "unidade", "receita", "despesa"}

MAX_VALOR_DIARIO = 1_000_000_000   # R$ 1 bilhão por dia — alerta acima disso
MIN_HISTORICO_DIAS = 60            # mínimo para gerar lags de 30d com margem
MAX_GAP_DIAS = 365                 # gap máximo tolerado entre datas consecutivas


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ResultadoValidacao:
    valido: bool = True
    erros: List[str] = field(default_factory=list)
    avisos: List[str] = field(default_factory=list)

    def adicionar_erro(self, msg: str) -> None:
        self.erros.append(msg)
        self.valido = False

    def adicionar_aviso(self, msg: str) -> None:
        self.avisos.append(msg)

    def imprimir(self) -> None:
        status = "VÁLIDO" if self.valido else "INVÁLIDO"
        print(f"\n[VALIDAÇÃO DE INPUT — {status}]")
        for e in self.erros:
            print(f"  [ERRO]   {e}")
        for a in self.avisos:
            print(f"  [AVISO]  {a}")
        if self.valido and not self.avisos:
            print("  Nenhum problema encontrado.")
        print()


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def _checar_colunas(df: pd.DataFrame, colunas: set, resultado: ResultadoValidacao) -> None:
    faltando = colunas - set(df.columns)
    if faltando:
        resultado.adicionar_erro(f"Colunas obrigatórias ausentes: {sorted(faltando)}")


def _checar_tipos(df: pd.DataFrame, resultado: ResultadoValidacao) -> None:
    for col in ["receita", "despesa"]:
        if col in df.columns:
            nao_numericos = pd.to_numeric(df[col], errors="coerce").isna().sum()
            if nao_numericos > 0:
                resultado.adicionar_erro(
                    f"Coluna '{col}' contém {nao_numericos} valores não numéricos."
                )


def _checar_datas(df: pd.DataFrame, resultado: ResultadoValidacao) -> None:
    if "ds" not in df.columns:
        return
    datas = pd.to_datetime(df["ds"], errors="coerce")
    nulas = datas.isna().sum()
    if nulas > 0:
        resultado.adicionar_erro(f"Coluna 'ds' contém {nulas} datas inválidas.")
        return

    # Datas no futuro
    hoje = pd.Timestamp.today().normalize()
    futuras = (datas > hoje).sum()
    if futuras > 0:
        resultado.adicionar_aviso(
            f"{futuras} datas estão no futuro — dados históricos não devem incluir datas futuras."
        )

    # Duplicatas
    duplicadas = datas.duplicated().sum()
    if duplicadas > 0:
        resultado.adicionar_aviso(f"{duplicadas} datas duplicadas encontradas.")

    # Gaps grandes
    datas_sorted = datas.sort_values().reset_index(drop=True)
    gaps = datas_sorted.diff().dt.days.dropna()
    gap_max = gaps.max()
    if gap_max > MAX_GAP_DIAS:
        resultado.adicionar_aviso(
            f"Gap de {int(gap_max)} dias entre datas consecutivas (máx tolerado: {MAX_GAP_DIAS})."
        )


def _checar_valores(df: pd.DataFrame, resultado: ResultadoValidacao) -> None:
    for col in ["receita", "despesa"]:
        if col not in df.columns:
            continue
        serie = pd.to_numeric(df[col], errors="coerce")

        # Valores negativos
        negativos = (serie < 0).sum()
        if negativos > 0:
            resultado.adicionar_erro(
                f"Coluna '{col}' contém {negativos} valores negativos — possível manipulação de input."
            )

        # Valores absurdamente altos
        absurdos = (serie > MAX_VALOR_DIARIO).sum()
        if absurdos > 0:
            resultado.adicionar_aviso(
                f"Coluna '{col}' contém {absurdos} valores acima de R$ {MAX_VALOR_DIARIO:,.0f} — verifique se são reais."
            )

        # Série completamente zerada
        if serie.sum() == 0:
            resultado.adicionar_erro(f"Coluna '{col}' está completamente zerada.")

        # Mais de 50% zeros (excluindo domingos seria ideal, mas aqui é uma heurística)
        pct_zero = (serie == 0).mean()
        if pct_zero > 0.5:
            resultado.adicionar_aviso(
                f"Coluna '{col}' tem {pct_zero*100:.1f}% de zeros — verifique se os dados estão completos."
            )


def _checar_historico(df: pd.DataFrame, resultado: ResultadoValidacao) -> None:
    if "ds" not in df.columns:
        return
    n_dias = df["ds"].nunique()
    if n_dias < MIN_HISTORICO_DIAS:
        resultado.adicionar_erro(
            f"Histórico com apenas {n_dias} dias. Mínimo recomendado: {MIN_HISTORICO_DIAS} dias."
        )


def _checar_nan(df: pd.DataFrame, resultado: ResultadoValidacao) -> None:
    for col in ["receita", "despesa"]:
        if col in df.columns:
            nulos = df[col].isna().sum()
            if nulos > 0:
                resultado.adicionar_aviso(
                    f"Coluna '{col}' contém {nulos} valores nulos — serão tratados pelo modelo."
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validar_base(df: pd.DataFrame, por_unidade: bool = False) -> ResultadoValidacao:
    """
    Valida o DataFrame de entrada antes do treinamento ou previsão.

    Parâmetros
    ----------
    df : DataFrame com colunas ds, receita, despesa (e opcionalmente unidade)
    por_unidade : se True, exige a coluna 'unidade'

    Retorna
    -------
    ResultadoValidacao com lista de erros e avisos
    """
    resultado = ResultadoValidacao()
    colunas = COLUNAS_UNIDADE if por_unidade else COLUNAS_OBRIGATORIAS

    _checar_colunas(df, colunas, resultado)

    if not resultado.valido:
        return resultado  # sem colunas não tem como continuar

    _checar_tipos(df, resultado)
    _checar_datas(df, resultado)
    _checar_valores(df, resultado)
    _checar_historico(df, resultado)
    _checar_nan(df, resultado)

    return resultado


def validar_ou_abortar(df: pd.DataFrame, por_unidade: bool = False) -> None:
    """Valida e lança exceção se houver erros críticos."""
    resultado = validar_base(df, por_unidade=por_unidade)
    resultado.imprimir()
    if not resultado.valido:
        raise ValueError(
            f"Validação de input falhou com {len(resultado.erros)} erro(s). "
            "Corrija os dados antes de treinar ou prever."
        )
