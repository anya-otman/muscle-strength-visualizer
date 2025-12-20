import re
from io import BytesIO
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


BASE_COLS = {"Дата осмотра", "ФИО", "Возраст"}

MRC_SCALE: Dict[int, str] = {
    5: "5 — Нормальная сила",
    4: "4 — Движение против небольшого сопротивления (с фиксацией до ~5 сек)",
    3: "3 — Движение против силы тяжести, но без сопротивления",
    2: "2 — Движения только в горизонтальной плоскости (без гравитации)",
    1: "1 — Слабое сокращение мышц",
    0: "0 — Движения отсутствуют",
}


# ----------------------------
# Утилиты
# ----------------------------
def _is_nan(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return x is None


def normalize_col_name(x) -> str:
    if _is_nan(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def make_unique_columns(cols: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, c in enumerate(cols):
        base = normalize_col_name(c)
        if base == "":
            base = f"Unnamed_{i}"
        n = seen.get(base, 0) + 1
        seen[base] = n
        out.append(base if n == 1 else f"{base} ({n})")
    return out


def strip_dup_suffix(col_name: str) -> str:
    return re.sub(r"\s\(\d+\)\s*$", "", str(col_name).strip())


def find_header_row(df_raw: pd.DataFrame) -> int:
    for i in range(min(len(df_raw), 80)):
        row = [normalize_col_name(x) for x in df_raw.iloc[i].tolist()]
        if "Дата осмотра" in row:
            return i
    raise ValueError("Не удалось найти строку заголовков (где есть 'Дата осмотра').")


def build_group_map_from_row(group_row: pd.Series, unique_columns: List[str]) -> Dict[str, str]:
    raw = list(group_row.values)
    if len(raw) < len(unique_columns):
        raw = raw + [None] * (len(unique_columns) - len(raw))
    elif len(raw) > len(unique_columns):
        raw = raw[: len(unique_columns)]

    current: Optional[str] = None
    out: Dict[str, str] = {}
    for col, g in zip(unique_columns, raw):
        if not _is_nan(g) and str(g).strip() != "":
            current = str(g).strip()
        out[col] = current if current else "—"
    return out


def is_pre_treatment_column(col_name: str) -> bool:
    base = strip_dup_suffix(col_name).lower()
    base = re.sub(r"\s+", " ", base).strip()
    return bool(re.search(r"\sдо\s*$", base))


def normalize_group_name(g: str) -> str:
    if g is None:
        return "—"
    s = str(g).strip().lower()
    if "бедр" in s:
        return "Бедро"
    if "голен" in s:
        return "Голень"
    if "стоп" in s or "ступн" in s:
        return "Стопа"
    return "—"


def strength_hover_text(strength: float) -> str:
    try:
        s_int = int(round(float(strength)))
    except Exception:
        return ""
    return MRC_SCALE.get(s_int, "")


# ----------------------------
# Чтение и подготовка данных
# ----------------------------
def load_patient_data_from_excel(file_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    df_raw = pd.read_excel(BytesIO(file_bytes), header=None)

    header_row_idx = find_header_row(df_raw)
    group_row_idx = header_row_idx - 1 if header_row_idx > 0 else None

    header = df_raw.iloc[header_row_idx].tolist()
    cols_norm = [normalize_col_name(x) for x in header]
    cols_unique = make_unique_columns(cols_norm)

    df = df_raw.iloc[header_row_idx + 1 :].copy()
    df.columns = cols_unique
    df = df.dropna(how="all").reset_index(drop=True)

    if group_row_idx is not None and group_row_idx >= 0:
        group_map = build_group_map_from_row(df_raw.iloc[group_row_idx], cols_unique)
    else:
        group_map = {c: "—" for c in cols_unique}

    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"В файле не найдены обязательные колонки: {missing}. "
            f"Найденные колонки: {list(df.columns)[:25]}..."
        )

    df["Дата осмотра"] = pd.to_datetime(df["Дата осмотра"], errors="coerce")
    if df["Дата осмотра"].isna().any():
        bad_rows = df[df["Дата осмотра"].isna()].index.tolist()
        raise ValueError(f"Есть строки с некорректной датой осмотра. Индексы строк: {bad_rows}")

    candidate_cols = [c for c in df.columns if c not in BASE_COLS]
    strength_cols = [c for c in candidate_cols if not is_pre_treatment_column(c)]
    if not strength_cols:
        raise ValueError("Не найдено ни одного столбца мышечной силы без суффикса 'до'.")

    for c in strength_cols:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if ((df[c] < 0) | (df[c] > 5)).any():
            bad = df.loc[(df[c] < 0) | (df[c] > 5), ["ФИО", "Дата осмотра", c]].head(10)
            raise ValueError(f"Найдены значения вне диапазона 0..5 в колонке '{c}'. Примеры:\n{bad}")

    df = df.sort_values(["ФИО", "Дата осмотра"]).reset_index(drop=True)

    group_map = {k: normalize_group_name(v) for k, v in group_map.items()}
    return df, group_map, strength_cols


def to_long_format(df_patient: pd.DataFrame, strength_cols: List[str], group_map: Dict[str, str]) -> pd.DataFrame:
    long_df = df_patient.melt(
        id_vars=["Дата осмотра", "ФИО", "Возраст"],
        value_vars=strength_cols,
        var_name="Движение",
        value_name="Сила",
    )
    long_df["Группа"] = long_df["Движение"].map(lambda x: group_map.get(x, "—"))
    long_df = long_df.sort_values(["Движение", "Дата осмотра"]).reset_index(drop=True)
    return long_df


# ----------------------------
# 2D график по движению
# ----------------------------
def plot_2d_per_movement(long_df: pd.DataFrame, movement: str) -> go.Figure:
    d = long_df[long_df["Движение"] == movement].copy().sort_values("Дата осмотра")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["Дата осмотра"],
            y=d["Сила"],
            mode="lines+markers",
            name=movement,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Сила: %{y}<br>"
                "%{text}"
                "<extra></extra>"
            ),
            text=[strength_hover_text(v) for v in d["Сила"]],
        )
    )

    group = d["Группа"].iloc[0] if len(d) else "—"
    fig.update_layout(
        title=f"{movement} (группа: {group})",
        xaxis_title="Дата осмотра",
        yaxis_title="Мышечная сила (0–5)",
        yaxis=dict(range=[-0.2, 5.2], dtick=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ----------------------------
# 3D Нога (улучшенная геометрия без внешних моделей)
# ----------------------------
def frustum_mesh(
    r_bottom: float,
    r_top: float,
    height: float,
    n: int = 96,
    z0: float = 0.0,
    bend_x: float = 0.0,
    bend_y: float = 0.0,
):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    xb = r_bottom * np.cos(theta)
    yb = r_bottom * np.sin(theta)
    xt = r_top * np.cos(theta)
    yt = r_top * np.sin(theta)

    shift_x_top = bend_x
    shift_y_top = bend_y

    x = np.concatenate([xb, xt + shift_x_top, [0.0], [shift_x_top]])
    y = np.concatenate([yb, yt + shift_y_top, [0.0], [shift_y_top]])
    z = np.concatenate([
        np.full(n, z0),
        np.full(n, z0 + height),
        [z0],
        [z0 + height],
    ])

    bottom_center_idx = 2 * n
    top_center_idx = 2 * n + 1

    faces_i, faces_j, faces_k = [], [], []

    for t in range(n):
        t_next = (t + 1) % n
        b0, b1 = t, t_next
        t0, t1 = t + n, t_next + n

        faces_i.append(b0); faces_j.append(b1); faces_k.append(t1)
        faces_i.append(b0); faces_j.append(t1); faces_k.append(t0)

    for t in range(n):
        t_next = (t + 1) % n
        faces_i.append(bottom_center_idx); faces_j.append(t_next); faces_k.append(t)

    for t in range(n):
        t_next = (t + 1) % n
        faces_i.append(top_center_idx); faces_j.append(t + n); faces_k.append(t_next + n)

    return x, y, z, np.array(faces_i), np.array(faces_j), np.array(faces_k)


def box_mesh(size_x: float, size_y: float, size_z: float, center: Tuple[float, float, float]):
    cx, cy, cz = center
    hx, hy, hz = size_x / 2.0, size_y / 2.0, size_z / 2.0

    verts = np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ])

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ])

    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    return x, y, z, i, j, k


def compute_group_means_for_date(long_df: pd.DataFrame, date_value: pd.Timestamp, movements: List[str]) -> Dict[str, float]:
    d = long_df[(long_df["Дата осмотра"] == date_value) & (long_df["Движение"].isin(movements))].copy()
    if d.empty:
        return {"Бедро": np.nan, "Голень": np.nan, "Стопа": np.nan}

    means = d.groupby("Группа")["Сила"].mean().to_dict()
    return {
        "Бедро": float(means.get("Бедро", np.nan)),
        "Голень": float(means.get("Голень", np.nan)),
        "Стопа": float(means.get("Стопа", np.nan)),
    }


def make_segment_trace(name: str, x, y, z, i, j, k, value: float, show_scale: bool) -> go.Mesh3d:
    intensity = np.full_like(x, fill_value=value, dtype=float)
    hover_text = f"{name}<br>Средняя сила: {value:.2f}" if not np.isnan(value) else f"{name}<br>Нет данных"
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=intensity,
        colorscale="Viridis",
        cmin=0, cmax=5,
        showscale=show_scale,
        colorbar=dict(title="Сила (0–5)") if show_scale else None,
        name=name,
        hovertemplate=f"{hover_text}<extra></extra>",
        opacity=0.97,
        flatshading=False,
    )


def _floor_mesh_trace(size: float, z: float, opacity: float = 0.14) -> go.Mesh3d:
    """
    Пол как Mesh3d из 2 треугольников (без hover), чтобы не появлялись направляющие.
    """
    # 4 вершины квадрата
    x = np.array([-size,  size,  size, -size], dtype=float)
    y = np.array([-size, -size,  size,  size], dtype=float)
    z_arr = np.array([z, z, z, z], dtype=float)

    # 2 треугольника: (0,1,2) и (0,2,3)
    i = np.array([0, 0], dtype=int)
    j = np.array([1, 2], dtype=int)
    k = np.array([2, 3], dtype=int)

    return go.Mesh3d(
        x=x, y=y, z=z_arr,
        i=i, j=j, k=k,
        color="rgba(160,160,160,1)",
        opacity=opacity,
        hoverinfo="skip",
        hovertemplate=None,
        showscale=False,
        name="floor",
    )


def plot_leg_3d_pretty(group_means: Dict[str, float], height_px: int = 520) -> go.Figure:
    thigh_h = 4.8
    shin_h = 4.2
    foot_x, foot_y, foot_z = 3.3, 1.55, 0.85
    gap = 0.25

    # стопа снизу
    z_foot_center = foot_z / 2.0
    x3, y3, z3, i3, j3, k3 = box_mesh(
        size_x=foot_x,
        size_y=foot_y,
        size_z=foot_z,
        center=(foot_x / 3.0, 0.05, z_foot_center),
    )

    # голень
    z_shin0 = foot_z + gap
    x2, y2, z2, i2, j2, k2 = frustum_mesh(
        r_bottom=1.05,
        r_top=1.18,
        height=shin_h,
        n=96,
        z0=z_shin0,
        bend_x=0.18,
        bend_y=0.10,
    )

    # бедро сверху
    z_thigh0 = z_shin0 + shin_h + gap
    x1, y1, z1, i1, j1, k1 = frustum_mesh(
        r_bottom=1.25,
        r_top=1.45,
        height=thigh_h,
        n=96,
        z0=z_thigh0,
        bend_x=-0.12,
        bend_y=0.06,
    )

    fig = go.Figure()
    fig.add_trace(make_segment_trace("Бедро", x1, y1, z1, i1, j1, k1, group_means.get("Бедро", np.nan), True))
    fig.add_trace(make_segment_trace("Голень", x2, y2, z2, i2, j2, k2, group_means.get("Голень", np.nan), False))
    fig.add_trace(make_segment_trace("Стопа", x3, y3, z3, i3, j3, k3, group_means.get("Стопа", np.nan), False))

    # Пол: Mesh3d, hover полностью отключён
    fig.add_trace(_floor_mesh_trace(size=7.0, z=-0.06, opacity=0.14))

    # Жёстко выключаем spikes (и делаем их прозрачными на всякий случай)
    axis_no_spikes = dict(
        visible=False,
        showspikes=False,
        spikethickness=0,
        spikecolor="rgba(0,0,0,0)",
        spikesides=False,
    )

    fig.update_layout(
        title="3D модель ноги",
        paper_bgcolor="#f6f7fb",
        scene=dict(
            bgcolor="#f6f7fb",
            xaxis=axis_no_spikes,
            yaxis=axis_no_spikes,
            zaxis=axis_no_spikes,
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.65, y=1.45, z=1.05),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=height_px,
        showlegend=False,
    )
    return fig


# ----------------------------
# Streamlit UI
# ----------------------------
def main() -> None:
    st.set_page_config(page_title="Динамика мышечной силы", layout="wide")
    st.title("Динамика мышечной силы")

    uploaded = st.file_uploader("Загрузите Excel файл (.xlsx)", type=["xlsx"])
    if not uploaded:
        st.info("Загрузите файл .xlsx, чтобы начать.")
        return

    try:
        df, group_map, strength_cols = load_patient_data_from_excel(uploaded.getvalue())
    except Exception as e:
        st.error(f"Ошибка чтения/валидации файла: {e}")
        return

    patients = sorted(df["ФИО"].astype(str).unique().tolist())
    selected_patient = st.selectbox("Выберите пациента", patients)

    df_patient = df[df["ФИО"].astype(str) == str(selected_patient)].copy()
    long_df = to_long_format(df_patient, strength_cols, group_map)

    all_movements = sorted(long_df["Движение"].unique().tolist())
    selected_movements = st.multiselect(
        "Какие движения учитывать (для 2D графиков и для окраски 3D модели)",
        options=all_movements,
        default=all_movements,
    )
    if not selected_movements:
        st.warning("Выберите хотя бы одно движение.")
        return

    tab2d, tab3d = st.tabs(["2D графики (каждое движение отдельно)", "3D модель ноги"])

    with tab2d:
        chosen = (
            long_df[long_df["Движение"].isin(selected_movements)][["Движение", "Группа"]]
            .drop_duplicates()
            .sort_values(["Группа", "Движение"])
        )
        for _, row in chosen.iterrows():
            mv = row["Движение"]
            st.plotly_chart(plot_2d_per_movement(long_df, mv), use_container_width=True)

    with tab3d:
        dates = sorted(long_df["Дата осмотра"].unique().tolist())
        if not dates:
            st.warning("Нет дат осмотра для выбранного пациента.")
            return

        date_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates]
        selected_label = st.select_slider(
            "Дата осмотра",
            options=date_labels,
            value=date_labels[-1],
        )
        selected_date = pd.to_datetime(selected_label)

        group_means = compute_group_means_for_date(long_df, selected_date, selected_movements)

        left, right = st.columns([1, 2], vertical_alignment="top")

        with left:
            st.metric("Бедро (среднее)", "—" if np.isnan(group_means["Бедро"]) else f'{group_means["Бедро"]:.2f}')
            st.metric("Голень (среднее)", "—" if np.isnan(group_means["Голень"]) else f'{group_means["Голень"]:.2f}')
            st.metric("Стопа (среднее)", "—" if np.isnan(group_means["Стопа"]) else f'{group_means["Стопа"]:.2f}')

        with right:
            st.plotly_chart(plot_leg_3d_pretty(group_means, height_px=520), use_container_width=True)


if __name__ == "__main__":
    main()
