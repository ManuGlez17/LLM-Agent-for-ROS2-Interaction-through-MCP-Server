import asyncio
import streamlit as st

from agent.react_agent import ReActAgent
from llm.client import LLMClient, uploaded_image_to_data_url
from mcp_client.ros_client_http import ROSMCPClient


st.set_page_config(
    page_title="ROS ReAct Agent",
    page_icon="🤖",
    layout="wide",
)


# ==========================================
# ESTADO DE STREAMLIT
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None


# ==========================================
# FUNCIÓN PARA EJECUTAR EL AGENTE
# ==========================================

async def execute_agent(user_message, image_url=None, ):

    llm = LLMClient()

    async with ROSMCPClient() as ros_mcp:

        tools = await ros_mcp.get_tools()

        agent = ReActAgent(
            llm=llm,
            mcp_client=ros_mcp,
            tools=tools,
        )

        # Recuperamos la conversación anterior
        agent.previous_response_id = (
            st.session_state.previous_response_id
        )

        response = await agent.run(
            goal = user_message,
            image_url=image_url,
        )

        # Guardamos el estado para el siguiente mensaje
        st.session_state.previous_response_id = (
            agent.previous_response_id
        )

        return response


# ==========================================
# INTERFAZ
# ==========================================

st.title("🤖 ROS ReAct Agent")

st.caption(
    "Agente LLM conectado a ROS2 mediante MCP"
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Estado")

    st.success("Interfaz activa")

    if st.session_state.previous_response_id:
        st.info("Conversación activa")
    else:
        st.info("Nueva conversación")

    st.divider()

    if st.button(
        "🗑️ Nueva conversación",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.session_state.previous_response_id = None

        st.rerun()


# ==========================================
# MOSTRAR HISTORIAL
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==========================================
# INPUT DEL USUARIO
# ==========================================

prompt = st.chat_input(
    "Escribe una instrucción para el robot...",
    accept_file=True,
    file_type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


if prompt:
    user_text = prompt.text

    uploaded_image = None

    if prompt.files:
        uploaded_image = prompt.files[0]
    else:
        uploaded_image = None
    
    if uploaded_image:
        st.image(
            uploaded_image,
            caption="Imagen enviada",
        )
    image_url = None

    if uploaded_image:

        image_url = uploaded_image_to_data_url(
            uploaded_image
        )

    if not user_text:
        user_text = "Describe esta imagen."
    # --------------------------------------
    # Mensaje usuario
    # --------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
    })

    with st.chat_message("user"):
        st.markdown(user_text)

    # --------------------------------------
    # Ejecutar agente
    # --------------------------------------

    with st.chat_message("assistant"):

        with st.status(
            "El agente está trabajando...",
            expanded=True,
        ) as status:

            try:

                st.write(
                    "🧠 Ejecutando ciclo ReAct..."
                )

                response = asyncio.run(
                    execute_agent(user_text, image_url=image_url)
                )

                status.update(
                    label="Tarea completada",
                    state="complete",
                    expanded=False,
                )

            except Exception as error:

                response = (
                    f"Error del agente: {error}"
                )

                status.update(
                    label="Error",
                    state="error",
                    expanded=True,
                )

        st.markdown(response)

    # --------------------------------------
    # Guardar respuesta
    # --------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
    })