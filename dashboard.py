from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="ROS Agent Experiments",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ROS MCP Agent — Experiment Dashboard")

st.caption(
    "Análisis de rendimiento del agente ReAct, LLM y herramientas MCP"
)


# ============================================================
# CARGA DE DATOS
# ============================================================

def find_episode_image(
    experiment_name,
    result_file,
    base_path,
):
    """
    Busca una imagen asociada al episodio.

    Ejemplos:
        episode_1.png
        episode_1.jpg
        episode_1.jpeg
        episode_1.webp
    """

    extensions = [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    ]

    result_file = Path(result_file)

    # -----------------------------------------
    # 1. Buscar junto al archivo de resultados
    # -----------------------------------------

    for extension in extensions:

        candidate = (
            result_file.parent
            / f"{experiment_name}{extension}"
        )

        if candidate.exists():
            return str(candidate)

    # -----------------------------------------
    # 2. Buscar recursivamente en results/
    # -----------------------------------------

    for extension in extensions:

        matches = list(
            base_path.rglob(
                f"{experiment_name}{extension}"
            )
        )

        if matches:
            return str(matches[0])

    return None

@st.cache_data
def load_experiments(base_directory: str):

    base_path = Path(base_directory)

    rows = []

    if not base_path.exists():
        return pd.DataFrame()

    # Buscar todos los JSONL dentro de la carpeta
    files = list(
        base_path.rglob("*.jsonl")
    )

    for file_path in files:

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as file:

                for line_number, line in enumerate(file, start=1):

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        record = json.loads(line)

                    except json.JSONDecodeError:
                        continue

                    data = record.get("data", {})

                    # Solo consideramos registros que parezcan
                    # resultados experimentales
                    if "task_time" not in data:
                        continue

                    experiment_name = record.get(
                        "experiment_name",
                        f"episode_{len(rows) + 1}",
                    )

                    image_path = find_episode_image(
                        experiment_name=experiment_name,
                        result_file=file_path,
                        base_path=base_path,
                    )

                    row = {
                        "source_file":
                            str(file_path),

                        "timestamp":
                            record.get("timestamp"),

                        "experiment_name":
                            experiment_name,

                        "image_path":
                            image_path,

                        "goal":
                            data.get("goal"),

                        "steps":
                            data.get("step", 0),

                        "final_answer":
                            data.get("final_answer"),

                        "task_time":
                            data.get("task_time", 0),

                        "llm_calls":
                            data.get("llm_calls", 0),

                        "tool_calls":
                            data.get("tool_calls", 0),

                        "llm_time":
                            data.get("llm_time", 0),

                        "tool_time":
                            data.get("tool_time", 0),

                        "input_tokens":
                            data.get("input_tokens", 0),

                        "output_tokens":
                            data.get("output_tokens", 0),

                        # Para cuando posteriormente
                        # agregues éxito explícito
                        "success":
                            data.get("success"),
                    }

                    rows.append(row)

        except Exception as error:

            st.warning(
                f"No se pudo leer {file_path}: {error}"
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ========================================================
    # MÉTRICAS DERIVADAS
    # ========================================================

    df["total_tokens"] = (
        df["input_tokens"]
        + df["output_tokens"]
    )

    df["avg_llm_latency"] = (
        df["llm_time"]
        / df["llm_calls"].replace(0, pd.NA)
    )

    df["avg_tool_latency"] = (
        df["tool_time"]
        / df["tool_calls"].replace(0, pd.NA)
    )

    df["overhead_time"] = (
        df["task_time"]
        - df["llm_time"]
        - df["tool_time"]
    )

    df["llm_time_percentage"] = (
        df["llm_time"]
        / df["task_time"].replace(0, pd.NA)
        * 100
    )

    df["tool_time_percentage"] = (
        df["tool_time"]
        / df["task_time"].replace(0, pd.NA)
        * 100
    )

    df["tokens_per_llm_call"] = (
        df["total_tokens"]
        / df["llm_calls"].replace(0, pd.NA)
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    return df


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Configuración")

    results_directory = st.text_input(
        "Carpeta de resultados",
        value="results",
    )

    if st.button(
        "🔄 Actualizar datos",
        use_container_width=True,
    ):

        st.cache_data.clear()
        st.rerun()


# ============================================================
# LEER DATOS
# ============================================================

df = load_experiments(
    results_directory
)


if df.empty:

    st.warning(
        "No se encontraron resultados experimentales."
    )

    st.stop()


# ============================================================
# FILTROS
# ============================================================

with st.sidebar:

    st.divider()

    st.header("Filtros")

    experiment_options = sorted(
        df["experiment_name"]
        .dropna()
        .unique()
    )

    selected_experiments = st.multiselect(
        "Experimentos",
        experiment_options,
        default=experiment_options,
    )


filtered_df = df[
    df["experiment_name"].isin(
        selected_experiments
    )
].copy()


if filtered_df.empty:

    st.warning(
        "No hay experimentos con los filtros seleccionados."
    )

    st.stop()


# ============================================================
# MÉTRICAS PRINCIPALES
# ============================================================

st.subheader("Resumen")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Experimentos",
    len(filtered_df),
)

col2.metric(
    "Tiempo medio de tarea",
    f"{filtered_df['task_time'].mean():.2f} s",
)

col3.metric(
    "Latencia media LLM",
    f"{filtered_df['avg_llm_latency'].mean():.2f} s",
)

col4.metric(
    "Tool calls promedio",
    f"{filtered_df['tool_calls'].mean():.1f}",
)


col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "LLM calls promedio",
    f"{filtered_df['llm_calls'].mean():.1f}",
)

col6.metric(
    "Pasos promedio",
    f"{filtered_df['steps'].mean():.1f}",
)

col7.metric(
    "Tokens promedio",
    f"{filtered_df['total_tokens'].mean():,.0f}",
)

col8.metric(
    "Tiempo MCP promedio",
    f"{filtered_df['tool_time'].mean():.2f} s",
)


# ============================================================
# SUCCESS RATE OPCIONAL
# ============================================================

if filtered_df["success"].notna().any():

    success_values = (
        filtered_df["success"]
        .dropna()
        .astype(bool)
    )

    success_rate = (
        success_values.mean() * 100
    )

    st.metric(
        "Success Rate",
        f"{success_rate:.1f} %",
    )


st.divider()


# ============================================================
# TABS
# ============================================================

tab_overview, tab_latency, tab_tokens, tab_detail, tab_data = (
    st.tabs(
        [
            "📊 Rendimiento",
            "⏱️ Latencia",
            "🧠 Tokens",
            "🔎 Episodios",
            "📄 Datos",
        ]
    )
)


# ============================================================
# TAB 1 - RENDIMIENTO
# ============================================================

with tab_overview:

    st.subheader(
        "Tiempo total por experimento"
    )

    fig = px.bar(
        filtered_df,
        x="experiment_name",
        y="task_time",
        labels={
            "experiment_name": "Experimento",
            "task_time": "Tiempo (s)",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.subheader(
        "Descomposición del tiempo"
    )

    time_df = filtered_df[
        [
            "experiment_name",
            "llm_time",
            "tool_time",
            "overhead_time",
        ]
    ].copy()

    time_df = time_df.melt(
        id_vars="experiment_name",
        var_name="component",
        value_name="seconds",
    )

    fig = px.bar(
        time_df,
        x="experiment_name",
        y="seconds",
        color="component",
        barmode="stack",
        labels={
            "experiment_name":
                "Experimento",

            "seconds":
                "Tiempo (s)",

            "component":
                "Componente",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.subheader(
        "Número de pasos"
    )

    fig = px.bar(
        filtered_df,
        x="experiment_name",
        y="steps",
        labels={
            "experiment_name":
                "Experimento",

            "steps":
                "Pasos",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# TAB 2 - LATENCIA
# ============================================================

with tab_latency:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Latencia media de decisión LLM"
        )

        fig = px.bar(
            filtered_df,
            x="experiment_name",
            y="avg_llm_latency",
            labels={
                "experiment_name":
                    "Experimento",

                "avg_llm_latency":
                    "Latencia (s)",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    with col2:

        st.subheader(
            "Latencia media de herramientas"
        )

        fig = px.bar(
            filtered_df,
            x="experiment_name",
            y="avg_tool_latency",
            labels={
                "experiment_name":
                    "Experimento",

                "avg_tool_latency":
                    "Latencia (s)",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    st.subheader(
        "LLM vs herramientas MCP"
    )

    fig = go.Figure()

    fig.add_bar(
        x=filtered_df["experiment_name"],
        y=filtered_df["llm_time"],
        name="LLM",
    )

    fig.add_bar(
        x=filtered_df["experiment_name"],
        y=filtered_df["tool_time"],
        name="MCP Tools",
    )

    fig.update_layout(
        barmode="group",
        xaxis_title="Experimento",
        yaxis_title="Tiempo (s)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.subheader(
        "Tiempo total vs número de llamadas LLM"
    )

    fig = px.scatter(
        filtered_df,
        x="llm_calls",
        y="task_time",
        hover_name="experiment_name",
        size="steps",
        labels={
            "llm_calls":
                "LLM calls",

            "task_time":
                "Tiempo total (s)",

            "steps":
                "Pasos",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# TAB 3 - TOKENS
# ============================================================

with tab_tokens:

    st.subheader(
        "Tokens por experimento"
    )

    token_df = filtered_df[
        [
            "experiment_name",
            "input_tokens",
            "output_tokens",
        ]
    ].melt(
        id_vars="experiment_name",
        var_name="type",
        value_name="tokens",
    )

    fig = px.bar(
        token_df,
        x="experiment_name",
        y="tokens",
        color="type",
        barmode="stack",
        labels={
            "experiment_name":
                "Experimento",

            "tokens":
                "Tokens",

            "type":
                "Tipo",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.subheader(
        "Tokens por llamada al LLM"
    )

    fig = px.bar(
        filtered_df,
        x="experiment_name",
        y="tokens_per_llm_call",
        labels={
            "experiment_name":
                "Experimento",

            "tokens_per_llm_call":
                "Tokens / llamada",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# TAB 4 - DETALLE POR EPISODIO
# ============================================================

with tab_detail:

    experiment = st.selectbox(
        "Selecciona un episodio",
        filtered_df["experiment_name"],
    )

    episode = (
        filtered_df[
            filtered_df["experiment_name"]
            == experiment
        ]
        .iloc[0]
    )

    st.subheader(
        experiment
    )

    info_col, image_col = st.columns(
        [1.2, 1]
    )

    with info_col:

        st.markdown("### Objetivo")

        st.write(
            episode["goal"]
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Tiempo",
            f"{episode['task_time']:.2f} s",
        )

        c2.metric(
            "Pasos",
            int(episode["steps"]),
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "LLM calls",
            int(episode["llm_calls"]),
        )

        c2.metric(
            "Tool calls",
            int(episode["tool_calls"]),
        )

        with image_col:

            st.markdown("### Resultado visual")

            image_path = episode.get(
                "image_path"
            )

            if (
                image_path
                and Path(image_path).exists()
            ):

                st.image(
                    image_path,
                    caption=experiment,
                    use_container_width=True,
                )

            else:

                st.info(
                    "No se encontró una imagen "
                    "para este episodio."
                )
                
    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latencia LLM",
        f"{episode['avg_llm_latency']:.2f} s",
    )

    c2.metric(
        "Latencia Tool",
        f"{episode['avg_tool_latency']:.2f} s",
    )

    c3.metric(
        "Input tokens",
        f"{int(episode['input_tokens']):,}",
    )

    c4.metric(
        "Output tokens",
        f"{int(episode['output_tokens']):,}",
    )


    st.markdown(
        "### Respuesta final"
    )

    st.write(
        episode["final_answer"]
    )


# ============================================================
# TAB 5 - DATOS COMPLETOS
# ============================================================

with tab_data:

    display_columns = [
        "experiment_name",
        "timestamp",
        "task_time",
        "steps",
        "llm_calls",
        "tool_calls",
        "avg_llm_latency",
        "avg_tool_latency",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]

    st.dataframe(
        filtered_df[
            display_columns
        ],
        use_container_width=True,
        hide_index=True,
    )


    # Exportación
    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Descargar CSV",
        data=csv,
        file_name="experiment_results.csv",
        mime="text/csv",
    )