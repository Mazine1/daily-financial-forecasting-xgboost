"""Envio das previsões ao banco de dados via variáveis de ambiente."""

import os
from datetime import date

import pandas as pd
from sqlalchemy import text

from extracao.db_connect import obter_engine

SCHEMA = os.getenv("DB_SCHEMA", "public")
TABELA = os.getenv("DB_TABELA_PREVISAO", "previsoes_diarias")


def _criar_tabela_se_necessario() -> None:
    engine = obter_engine()
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABELA} (
        id           SERIAL PRIMARY KEY,
        data_prevista DATE NOT NULL,
        unidade       TEXT,
        receita_prev  NUMERIC(14, 2),
        despesa_prev  NUMERIC(14, 2),
        saldo_prev    NUMERIC(14, 2),
        gerado_em     TIMESTAMP DEFAULT NOW()
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print(f"[OK] Tabela {SCHEMA}.{TABELA} verificada/criada.")


def salvar_previsao_db(
    df: pd.DataFrame,
    data_inicio: date | None = None,
    data_fim: date | None = None,
) -> None:
    """
    Envia previsões ao banco. Remove registros do intervalo antes de inserir
    para evitar duplicatas.
    """
    _criar_tabela_se_necessario()

    inicio = data_inicio or date.today()
    engine = obter_engine()

    df_db = df.copy()
    df_db["data_prevista"] = pd.to_datetime(df_db["ds"]).dt.date
    colunas = ["data_prevista", "unidade", "receita_prev", "despesa_prev", "saldo_prev"]
    colunas_existentes = [c for c in colunas if c in df_db.columns]
    df_db = df_db[colunas_existentes]

    with engine.begin() as conn:
        if data_fim:
            conn.execute(
                text(f"DELETE FROM {SCHEMA}.{TABELA} WHERE data_prevista >= :inicio AND data_prevista <= :fim"),
                {"inicio": inicio, "fim": data_fim},
            )
        else:
            conn.execute(
                text(f"DELETE FROM {SCHEMA}.{TABELA} WHERE data_prevista >= :inicio"),
                {"inicio": inicio},
            )

    with engine.begin() as conn:
        df_db.to_sql(TABELA, conn, schema=SCHEMA, if_exists="append", index=False)

    print(f"[OK] {len(df_db)} linhas de previsão gravadas em {SCHEMA}.{TABELA}.")
