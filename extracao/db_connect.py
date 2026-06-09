"""Conexão com banco de dados via variáveis de ambiente."""

import os
from functools import lru_cache

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def obter_engine():
    """Cria engine SQLAlchemy a partir das variáveis de ambiente."""
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT", "5432"))
    db = os.getenv("DB_NAME")
    usuario = os.getenv("DB_USER")
    senha = os.getenv("DB_PASS")

    if not all([host, db, usuario, senha]):
        raise EnvironmentError(
            "Variáveis de ambiente de banco incompletas. "
            "Configure DB_HOST, DB_NAME, DB_USER e DB_PASS no arquivo .env"
        )

    url = f"postgresql+psycopg2://{usuario}:{senha}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)


def testar_conexao() -> bool:
    """Testa se a conexão com o banco está funcionando."""
    try:
        engine = obter_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[OK] Conexão com banco de dados estabelecida com sucesso.")
        return True
    except Exception as e:
        print(f"[ERRO] Falha ao conectar ao banco: {e}")
        return False
