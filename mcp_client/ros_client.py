from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ROSMCPClient:

    def __init__(self):

        self.server_params = StdioServerParameters(
            command="uvx",
            args=[
                "ros-mcp",
                "--transport=stdio",
            ],
        )

        self.session = None
        self._stdio_context = None
        self._session_context = None

    async def __aenter__(self):

        # Iniciar proceso ros-mcp-server
        self._stdio_context = stdio_client(
            self.server_params
        )

        read, write = await self._stdio_context.__aenter__()

        # Crear sesión MCP
        self._session_context = ClientSession(
            read,
            write,
        )

        self.session = await self._session_context.__aenter__()

        # Handshake MCP
        await self.session.initialize()

        print("Sesión MCP inicializada.")

        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        if self._session_context is not None:

            await self._session_context.__aexit__(
                exc_type,
                exc_value,
                traceback,
            )

        if self._stdio_context is not None:

            await self._stdio_context.__aexit__(
                exc_type,
                exc_value,
                traceback,
            )

    async def get_tools(self):

        if self.session is None:
            raise RuntimeError(
                "La sesión MCP no está inicializada. "
                "Usa 'async with ROSMCPClient() as ros_mcp'."
            )

        result = await self.session.list_tools()

        tools = []

        for tool in result.tools:

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

    async def call_tool(
        self,
        name,
        arguments,
    ):

        if self.session is None:
            raise RuntimeError(
                "La sesión MCP no está inicializada."
            )

        return await self.session.call_tool(
            name,
            arguments=arguments,
        )