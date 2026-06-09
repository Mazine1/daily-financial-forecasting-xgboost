"""Treinamento de modelos XGBoost com validação temporal por unidade."""

import inspect
import os
import pickle
import random
from datetime import date, timedelta
from statistics import mean
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

LAGS = [1, 7, 14, 30, 365]
ROLLS = [7, 30]
GROWTH_WINDOWS = [7, 30]
FERIADOS_LOCAIS_PATH = "dados/feriados_locais.csv"

CALENDAR_COLS = [
    "ds_num", "dia", "mes", "ano", "dia_semana", "dia_ano", "semana_mes",
    "fim_mes", "inicio_mes", "dias_para_fim_mes",
    "janela_inicio_mes_5d", "janela_fim_mes_5d",
    "feriado_nacional", "prox_feriado_7d", "pos_feriado_7d",
    "feriado_local", "prox_feriado_local_7d", "pos_feriado_local_7d",
]

N_SPLITS = 4
TEST_SIZE = 30
MAX_CANDIDATOS = 6
RANDOM_STATE = 42
OUTLIER_Q_LOW = 0.005
OUTLIER_Q_HIGH = 0.995
EARLY_STOPPING_ROUNDS = 50
MIN_TRAIN_SAMPLES = 45
MIN_TEST_SIZE = 7
MIN_SPLITS = 2

BASE_PARAMS = dict(
    n_estimators=1800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    min_child_weight=2.0,
    reg_alpha=0.1,
    reg_lambda=1.5,
    gamma=0.0,
    eval_metric="mae",
    random_state=RANDOM_STATE,
)

CANDIDATOS_EXTRA = [
    {"max_depth": 5, "min_child_weight": 1.5, "subsample": 0.9, "colsample_bytree": 0.9, "learning_rate": 0.04, "n_estimators": 2000, "reg_alpha": 0.05, "reg_lambda": 1.2, "gamma": 0.0},
    {"max_depth": 7, "min_child_weight": 3.0, "subsample": 0.9, "colsample_bytree": 0.85, "learning_rate": 0.05, "n_estimators": 1600, "reg_alpha": 0.2, "reg_lambda": 2.0, "gamma": 0.05},
    {"max_depth": 6, "min_child_weight": 1.0, "subsample": 0.95, "colsample_bytree": 0.9, "learning_rate": 0.06, "n_estimators": 1500, "reg_alpha": 0.05, "reg_lambda": 1.0, "gamma": 0.0},
    {"max_depth": 5, "min_child_weight": 4.0, "subsample": 0.85, "colsample_bytree": 0.8, "learning_rate": 0.04, "n_estimators": 2200, "reg_alpha": 0.3, "reg_lambda": 2.5, "gamma": 0.1},
    {"max_depth": 7, "min_child_weight": 2.0, "subsample": 0.85, "colsample_bytree": 0.9, "learning_rate": 0.03, "n_estimators": 2400, "reg_alpha": 0.1, "reg_lambda": 1.8, "gamma": 0.0},
]


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------

def _ajustar_janelas_por_amostra(n_rows: int) -> Tuple[List[int], List[int], List[int]]:
    max_window = max(1, n_rows - MIN_TRAIN_SAMPLES)
    lags = [lag for lag in LAGS if lag <= max_window] or [1]
    rolls = [j for j in ROLLS if j <= max_window]
    growth_windows = [j for j in GROWTH_WINDOWS if j <= max_window]
    return lags, rolls, growth_windows


def _definir_tscv(n_samples: int) -> TimeSeriesSplit | None:
    if n_samples < MIN_TEST_SIZE * (MIN_SPLITS + 1):
        return None
    test_size = min(TEST_SIZE, max(MIN_TEST_SIZE, n_samples // (N_SPLITS + 1)))
    max_splits = (n_samples // test_size) - 1
    n_splits = min(N_SPLITS, max_splits)
    if n_splits < MIN_SPLITS:
        return None
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size)


# ---------------------------------------------------------------------------
# Feriados
# ---------------------------------------------------------------------------

def _pascoa(ano: int) -> date:
    """Algoritmo de Butcher para data da Páscoa."""
    a = ano % 19
    b, c = divmod(ano, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    return date(ano, mes, dia)


def _feriados_nacionais(anos: List[int]) -> List[date]:
    feriados: set[date] = set()
    for ano in anos:
        p = _pascoa(ano)
        feriados.update({
            date(ano, 1, 1), date(ano, 4, 21), date(ano, 5, 1),
            date(ano, 9, 7), date(ano, 10, 12), date(ano, 11, 2),
            date(ano, 11, 15), date(ano, 12, 25),
            p, p - timedelta(days=2), p - timedelta(days=47),
            p + timedelta(days=60),
        })
    return sorted(feriados)


def _feriados_locais(anos: List[int], caminho: str = FERIADOS_LOCAIS_PATH) -> List[date]:
    """Carrega feriados locais do CSV (colunas: data, cidade opcional, descricao)."""
    datas: set[date] = set()
    if not os.path.exists(caminho):
        return []
    try:
        df = pd.read_csv(caminho)
    except Exception:
        return []
    if "data" not in df.columns:
        return []

    for _, row in df.iterrows():
        raw = str(row.get("data", "")).strip()
        if not raw:
            continue
        partes = raw.split("-")
        if len(partes) == 2:
            for ano in anos:
                try:
                    datas.add(date(ano, int(partes[0]), int(partes[1])))
                except ValueError:
                    pass
        elif len(partes) == 3:
            try:
                datas.add(date(int(partes[0]), int(partes[1]), int(partes[2])))
            except ValueError:
                pass
    return sorted(datas)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _flags_janela(datas: pd.Series, feriados: List[date]) -> Tuple[List[int], List[int]]:
    if not feriados:
        return [0] * len(datas), [0] * len(datas)
    fs = sorted(feriados)
    prox, pos = [], []
    for d in datas.dt.date:
        dist_prox = min(((f - d).days for f in fs if f >= d), default=999)
        dist_pos = min(((d - f).days for f in fs if f <= d), default=999)
        prox.append(1 if 0 <= dist_prox <= 7 else 0)
        pos.append(1 if 0 <= dist_pos <= 7 else 0)
    return prox, pos


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ds_num"] = df["ds"].astype("int64") // 10**9
    df["dia"] = df["ds"].dt.day
    df["mes"] = df["ds"].dt.month
    df["ano"] = df["ds"].dt.year
    df["dia_semana"] = df["ds"].dt.weekday
    df["dia_ano"] = df["ds"].dt.dayofyear
    df["semana_mes"] = ((df["ds"].dt.day - 1) // 7) + 1
    df["fim_mes"] = df["ds"].dt.is_month_end.astype(int)
    df["inicio_mes"] = df["ds"].dt.is_month_start.astype(int)
    df["dias_para_fim_mes"] = df["ds"].dt.days_in_month - df["ds"].dt.day
    df["janela_inicio_mes_5d"] = (df["ds"].dt.day <= 5).astype(int)
    df["janela_fim_mes_5d"] = (df["dias_para_fim_mes"] <= 4).astype(int)

    anos = df["ds"].dt.year.unique().tolist()
    fer_nac = _feriados_nacionais(anos)
    fer_loc = _feriados_locais(anos)

    df["feriado_nacional"] = df["ds"].dt.date.isin(fer_nac).astype(int)
    prox, pos = _flags_janela(df["ds"], fer_nac)
    df["prox_feriado_7d"], df["pos_feriado_7d"] = prox, pos

    df["feriado_local"] = df["ds"].dt.date.isin(fer_loc).astype(int)
    prox_l, pos_l = _flags_janela(df["ds"], fer_loc)
    df["prox_feriado_local_7d"], df["pos_feriado_local_7d"] = prox_l, pos_l
    return df


def add_target_lags(
    df: pd.DataFrame,
    alvo: str,
    lags: List[int],
    rolls: List[int],
    growth_windows: List[int],
) -> pd.DataFrame:
    df = df.copy()
    s = pd.to_numeric(df[alvo], errors="coerce")
    df[alvo] = s
    for lag in lags:
        df[f"{alvo}_lag{lag}"] = s.shift(lag)
    for j in rolls:
        df[f"{alvo}_roll{j}"] = s.shift(1).rolling(j).mean()
    for j in growth_windows:
        atual = s.shift(1)
        ref = s.shift(1 + j)
        df[f"{alvo}_crescimento_{j}d"] = (atual - ref) / (ref.abs() + 1e-6)
        df[f"{alvo}_tend_{j}d"] = s.shift(1).rolling(j).mean() - s.shift(1 + j).rolling(j).mean()
    return df


def clip_outliers(
    serie: pd.Series,
    lower_q: float = OUTLIER_Q_LOW,
    upper_q: float = OUTLIER_Q_HIGH,
) -> Tuple[pd.Series, Tuple[float, float]]:
    s = pd.to_numeric(serie, errors="coerce")
    q_low = s.quantile(lower_q)
    q_high = s.quantile(upper_q)
    if pd.notna(s.min(skipna=True)) and s.min(skipna=True) >= 0 and (s == 0).any():
        q_low = 0.0
    return s.clip(lower=q_low, upper=q_high), (float(q_low), float(q_high))


def preparar_dados(
    df: pd.DataFrame,
    alvo: str,
    lags: List[int],
    rolls: List[int],
    growth_windows: List[int],
) -> Tuple[pd.DataFrame, pd.Series, List[str], Tuple[float, float]]:
    df = df.sort_values("ds").copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df[alvo], caps = clip_outliers(df[alvo])
    df = add_calendar_features(df)
    df = add_target_lags(df, alvo, lags, rolls, growth_windows)
    df = df.dropna()
    if df.empty:
        raise ValueError(f"Dados insuficientes para treinar '{alvo}' após gerar lags/rollings.")

    features = (
        CALENDAR_COLS
        + [f"{alvo}_lag{lag}" for lag in lags]
        + [f"{alvo}_roll{j}" for j in rolls]
        + [f"{alvo}_crescimento_{j}d" for j in growth_windows]
        + [f"{alvo}_tend_{j}d" for j in growth_windows]
    )
    return df[features], df[alvo], features, caps


# ---------------------------------------------------------------------------
# Avaliação e seleção de hiperparâmetros
# ---------------------------------------------------------------------------

def _avaliar_candidato(
    X: pd.DataFrame, y: pd.Series, params: Dict, tscv: TimeSeriesSplit
) -> Tuple[float, int]:
    maes, best_rounds = [], []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = XGBRegressor(**params)
        fit_kwargs: Dict = {"eval_set": [(X_val, y_val)], "verbose": False}
        try:
            sig = inspect.signature(model.fit)
            if "early_stopping_rounds" in sig.parameters:
                fit_kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        except (TypeError, ValueError):
            pass
        try:
            model.fit(X_tr, y_tr, **fit_kwargs)
        except TypeError as exc:
            if "early_stopping_rounds" in str(exc):
                fit_kwargs.pop("early_stopping_rounds", None)
                model.fit(X_tr, y_tr, **fit_kwargs)
            else:
                raise
        maes.append(mean_absolute_error(y_val, model.predict(X_val)))
        best_iter = getattr(model, "best_iteration", None)
        best_rounds.append(int(best_iter + 1) if best_iter is not None else model.n_estimators)
    return mean(maes), int(mean(best_rounds))


# ---------------------------------------------------------------------------
# Relatório de overfitting
# ---------------------------------------------------------------------------

def _relatorio_overfitting(
    modelo: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    alvo: str,
    janela_teste: int = 30,
) -> None:
    """
    Compara MAE no treino vs MAE nos últimos 30 dias (nunca vistos no treino final).
    Alerta se o modelo memorizou os dados em vez de generalizar.
    """
    if len(X) <= janela_teste:
        return

    X_treino, X_teste = X.iloc[:-janela_teste], X.iloc[-janela_teste:]
    y_treino, y_teste = y.iloc[:-janela_teste], y.iloc[-janela_teste:]

    mae_treino = mean_absolute_error(y_treino, modelo.predict(X_treino))
    mae_teste  = mean_absolute_error(y_teste,  modelo.predict(X_teste))
    ratio      = mae_teste / mae_treino if mae_treino > 0 else float("inf")

    print(f"   [OVERFITTING] {alvo.upper()}: MAE treino={mae_treino:,.0f} | MAE teste={mae_teste:,.0f} | ratio={ratio:.2f}")

    if ratio > 3.0:
        print(f"   [ALERTA] Ratio > 3.0 — possível overfitting. Considere mais regularização.")
    elif ratio > 1.5:
        print(f"   [AVISO]  Ratio > 1.5 — generalização moderada.")
    else:
        print(f"   [OK]     Modelo generaliza bem (ratio <= 1.5).")


# ---------------------------------------------------------------------------
# Treinamento principal
# ---------------------------------------------------------------------------

def treinar_modelo(
    df: pd.DataFrame,
    alvo: str,
    nome_modelo: str,
    lags: List[int] | None = None,
    rolls: List[int] | None = None,
    growth_windows: List[int] | None = None,
) -> None:
    print(f"\n--> Treinando modelo XGBoost para {alvo.upper()}...")
    lags = lags or LAGS
    rolls = rolls or ROLLS
    growth_windows = growth_windows or GROWTH_WINDOWS

    X, y, features, caps = preparar_dados(df, alvo, lags, rolls, growth_windows)
    tscv = _definir_tscv(len(X))

    if tscv is None:
        print("   [AVISO] Histórico curto — treinando sem validação cruzada.")
        modelo_final = XGBRegressor(**BASE_PARAMS)
        modelo_final.fit(X, y)
    else:
        random.seed(RANDOM_STATE)
        candidatos = [BASE_PARAMS.copy()]
        for extra in random.sample(CANDIDATOS_EXTRA, k=min(MAX_CANDIDATOS - 1, len(CANDIDATOS_EXTRA))):
            p = BASE_PARAMS.copy()
            p.update(extra)
            candidatos.append(p)

        melhor: Dict | None = None
        for idx, params in enumerate(candidatos, start=1):
            mae_medio, n_est = _avaliar_candidato(X, y, params, tscv)
            print(f"   Candidato {idx}: MAE={mae_medio:.2f} | n_estimators={n_est}")
            if melhor is None or mae_medio < melhor["mae"]:
                melhor = {"mae": mae_medio, "n_est": n_est, "params": params}

        print(f"   Melhor MAE médio: {melhor['mae']:.2f} | n_estimators: {melhor['n_est']}")
        params_finais = {**melhor["params"], "n_estimators": melhor["n_est"]}
        modelo_final = XGBRegressor(**params_finais)
        modelo_final.fit(X, y)

    # Relatório de overfitting: MAE treino vs MAE teste (últimos 30 dias)
    _relatorio_overfitting(modelo_final, X, y, alvo)

    os.makedirs("modelos", exist_ok=True)
    caminho = f"modelos/{nome_modelo}.pkl"
    with open(caminho, "wb") as f:
        pickle.dump(
            {
                "model": modelo_final,
                "features": features,
                "lags": lags,
                "rolls": rolls,
                "calendar_cols": CALENDAR_COLS,
                "growth_windows": growth_windows,
                "outlier_caps": caps,
            },
            f,
        )
    print(f"   [OK] Modelo salvo em {caminho}")


def treinar_todos_por_unidade(df: pd.DataFrame, unidades: List[str]) -> None:
    if "unidade" not in df.columns:
        raise ValueError("Coluna 'unidade' não encontrada para treino por unidade.")

    pad = max(len(str(u)) for u in unidades) if unidades else 1
    df = df.copy()
    df["unidade"] = df["unidade"].astype(str).str.zfill(pad)

    for unidade in unidades:
        u = str(unidade).zfill(pad)
        df_u = df[df["unidade"] == u].copy()
        if df_u.empty:
            print(f"[AVISO] Unidade {u} sem dados. Pulando.")
            continue
        print(f"\n=== Unidade {u} ===")
        try:
            lags, rolls, gw = _ajustar_janelas_por_amostra(len(df_u))
            treinar_modelo(df_u, "receita", f"xgb_receita_{u}", lags, rolls, gw)
            treinar_modelo(df_u, "despesa", f"xgb_despesa_{u}", lags, rolls, gw)
        except ValueError as exc:
            print(f"[AVISO] Unidade {u}: {exc}. Pulando.")

    print("\n[OK] TODOS OS MODELOS TREINADOS!")
