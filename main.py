import asyncio
import json

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from openai import AsyncOpenAI


load_dotenv()


MODEL = "gpt-5.6-terra"


AGENT_INSTRUCTIONS = """
Eres un agente encargado de interactuar con un robot mediante ROS.

Tienes acceso a herramientas de ROS a través de un servidor MCP.

Reglas:

1. Usa las herramientas MCP para obtener información real del sistema ROS.
2. Nunca inventes tópicos, nodos, servicios ni resultados.
3. Si necesitas conocer el estado de ROS antes de actuar, consulta primero
   las herramientas disponibles.
4. Antes de realizar movimientos, asegúrate de conocer el tópico,
   tipo de mensaje y estructura necesaria.
5. Usa velocidades pequeñas para pruebas.
6. Después de una acción importante, verifica el resultado cuando sea posible.
7. Si una herramienta devuelve un error, analiza el error antes de volver
   a intentarlo.
8. No ejecutes acciones innecesarias.
"""


def convert_mcp_tools(mcp_tools):
    """
    Convierte las herramientas que proporciona MCP
    al formato de Function Calling del LLM.
    """

    tools = []

    for tool in mcp_tools:

        tools.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
                "strict": False,
            }
        )

    return tools


def serialize_mcp_result(result):
    """
    Convierte el resultado MCP a texto para devolvérselo al LLM.
    """

    data = {
        "is_error": result.is_error,
    }

    if result.structured_content is not None:
        data["structured_content"] = result.structured_content

    if result.content:
        content = []

        for block in result.content:

            if hasattr(block, "text"):
                content.append(block.text)

            elif hasattr(block, "model_dump"):
                content.append(
                    block.model_dump(mode="json")
                )

            else:
                content.append(str(block))

        data["content"] = content

    return json.dumps(
        data,
        ensure_ascii=False,
        default=str,
    )


async def run_agent(
    llm,
    mcp_client,
    tools,
    user_message,
    previous_response_id=None,
):

    request = {
        "model": MODEL,
        "instructions": AGENT_INSTRUCTIONS,
        "input": user_message,
        "tools": tools,
    }

    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id

    response = await llm.responses.create(**request)

    # Evita que el agente pueda entrar en un bucle infinito
    MAX_STEPS = 10

    for step in range(MAX_STEPS):

        print(f"\n--- Paso del agente {step + 1} ---")

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        # Si no pide herramientas, terminó
        if not tool_calls:
            return response.output_text, response.id

        tool_outputs = []

        for call in tool_calls:

            print(f"Tool seleccionada: {call.name}")
            print(f"Argumentos: {call.arguments}")

            try:

                arguments = json.loads(call.arguments)

                result = await mcp_client.call_tool(
                    call.name,
                    arguments,
                )

                result_text = serialize_mcp_result(result)

                print("Resultado:")
                print(result_text)

            except Exception as error:

                result_text = json.dumps(
                    {
                        "error": str(error)
                    },
                    ensure_ascii=False,
                )

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result_text,
                }
            )

        # Devolver las observaciones al LLM
        response = await llm.responses.create(
            model=MODEL,
            instructions=AGENT_INSTRUCTIONS,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=tools,
        )

    return (
        "El agente alcanzó el límite máximo de pasos.",
        response.id,
    )


async def main():

    # ============================
    # LLM
    # ============================

    llm = AsyncOpenAI()

    # ============================
    # ROS MCP Server
    # ============================

    server = StdioServerParameters(
        command="uvx",
        args=[
            "ros-mcp",
            "--transport=stdio",
        ],
    )

    async with Client(server) as mcp_client:

        print("Conectado a ros-mcp-server.")

        # ============================
        # Descubrir herramientas MCP
        # ============================

        tool_result = await mcp_client.list_tools()

        print("\nHerramientas MCP encontradas:")

        for tool in tool_result.tools:
            print(f" - {tool.name}")

        # ============================
        # Convertir MCP -> LLM
        # ============================

        llm_tools = convert_mcp_tools(
            tool_result.tools
        )

        print(
            f"\n{len(llm_tools)} herramientas "
            "disponibles para el LLM."
        )

        # ============================
        # Chat
        # ============================

        previous_response_id = None

        print("\nAgente listo.")
        print("Escribe 'salir' para terminar.\n")

        while True:

            user_message = input("Tú: ")

            if user_message.lower() in {
                "salir",
                "exit",
                "quit",
            }:
                break

            answer, previous_response_id = await run_agent(
                llm=llm,
                mcp_client=mcp_client,
                tools=llm_tools,
                user_message=user_message,
                previous_response_id=previous_response_id,
            )

            print(f"\nAgente: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
