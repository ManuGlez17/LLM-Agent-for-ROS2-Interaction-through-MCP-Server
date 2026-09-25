import asyncio

from agent.react_agent import ReActAgent
from llm.client import LLMClient
from mcp_client.ros_client import ROSMCPClient


async def main():

    print("Iniciando agente...")

    llm = LLMClient()

    async with ROSMCPClient() as ros_mcp:

        print("Conectado a ros-mcp-server.")

        tools = await ros_mcp.get_tools()

        print(
            f"\n{len(tools)} herramientas MCP encontradas:"
        )

        for tool in tools:
            print(f" - {tool['name']}")

        agent = ReActAgent(
            llm=llm,
            mcp_client=ros_mcp,
            tools=tools,
        )

        print("\nAgente listo.")
        print("Escribe 'salir' para terminar.\n")

        while True:

            user_input = input("Tú: ").strip()

            if not user_input:
                continue

            if user_input.lower() in {
                "salir",
                "exit",
                "quit",
            }:
                break

            try:

                response = await agent.run(
                    user_input
                )

                print(
                    f"\nAgente: {response}\n"
                )

            except Exception as error:

                print(
                    f"\nError: {error}\n"
                )


if __name__ == "__main__":
    asyncio.run(main())