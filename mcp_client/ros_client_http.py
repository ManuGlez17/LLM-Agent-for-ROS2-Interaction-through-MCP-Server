from mcp import Client


class ROSMCPClient:

    def __init__(
        self,
        url="http://127.0.0.1:9000/mcp",
    ):
        self.url = url
        self.client = None
        self._context = None

    async def __aenter__(self):

        self._context = Client(
            self.url
        )

        self.client = (
            await self._context.__aenter__()
        )

        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        await self._context.__aexit__(
            exc_type,
            exc_value,
            traceback,
        )

    async def get_tools(self):

        result = await self.client.list_tools()

        return [
            {
                "type": "function",
                "name": tool.name,
                "description":
                    tool.description or "",
                "parameters":
                    tool.input_schema,
                "strict": False,
            }
            for tool in result.tools
        ]

    async def call_tool(
        self,
        name,
        arguments,
    ):

        return await self.client.call_tool(
            name,
            arguments,
        )