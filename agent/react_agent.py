import json

from .states import AgentState
from .context import AgentContext


class ReActAgent:

    def __init__(
        self,
        llm,
        mcp_client,
        tools,
        model="gpt-5.6-terra",
    ):

        self.llm = llm
        self.mcp = mcp_client

        self.tools = tools
        self.model = model

        self.max_steps = 100

    async def run(self, goal: str):

        context = AgentContext(
            goal=goal
        )
        
        while context.state not in {
            AgentState.DONE,
            AgentState.ERROR,
        }:

            context.step += 1

            print(
                f"\n[{context.step}] "
                f"STATE = {context.state.name}"
            )

            if context.step > self.max_steps:

                context.state = AgentState.ERROR
                context.final_answer = (
                    "Se alcanzó el número máximo de pasos."
                )

                break

            if context.state == AgentState.DECIDE:

                await self._decide(context)

            elif context.state == AgentState.ACT:

                await self._act(context)

            elif context.state == AgentState.OBSERVE:

                await self._observe(context)

        return context.final_answer

    async def _decide(self, context):

        print("LLM decidiendo siguiente acción...")

        instructions = """
    Eres un agente encargado de controlar e inspeccionar
    un sistema ROS utilizando herramientas MCP.

    Tu objetivo es completar la solicitud del usuario.

    Reglas:

    - Utiliza herramientas MCP cuando necesites información
    del sistema ROS o ejecutar una acción.
    - Nunca inventes resultados de ROS.
    - Realiza una sola acción importante a la vez.
    - Después de recibir el resultado de una herramienta,
    utiliza esa observación para decidir qué hacer después.
    - Si el objetivo ya está cumplido, responde al usuario.
    """

        # Primera iteración
        if context.previous_response_id is None:

            input_data = context.goal

            response = await self.llm.generate(
                instructions=instructions,
                input_data=input_data,
                tools=self.tools,
            )

        # Hay una observación de la acción anterior
        else:

            input_data = [
                {
                    "type": "function_call_output",
                    "call_id":
                        context.pending_call.call_id,

                    "output":
                        context.observation,
                }
            ]


            response = await self.llm.generate(
                instructions=instructions,
                previous_response_id=
                    context.previous_response_id,
                input_data=input_data,
                tools=self.tools,
            )

        context.previous_response_id = response.id

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if tool_calls:

            context.pending_call = tool_calls[0]

            print(
                "Acción seleccionada:",
                context.pending_call.name
            )

            context.state = AgentState.ACT

            return

        context.final_answer = response.output_text
        context.state = AgentState.DONE

    async def _act(self, context):

        call = context.pending_call

        print(f"Ejecutando MCP: {call.name}")

        try:

            arguments = json.loads(
                call.arguments
            )

            print(
                "Argumentos:",
                json.dumps(
                    arguments,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            result = await self.mcp.call_tool(
                call.name,
                arguments,
            )

            context.observation = (
                self._serialize_result(result)
            )

            context.history.append(
                {
                    "step": context.step,
                    "action": call.name,
                    "arguments": arguments,
                }
            )

            context.state = AgentState.OBSERVE

        except Exception as error:

            context.observation = json.dumps({
                "error": str(error)
            })

            context.state = AgentState.OBSERVE

    async def _observe(self, context):

        print("Observación recibida:")

        print(context.observation)

        context.history.append(
            {
                "step": context.step,
                "observation":
                    context.observation,
            }
        )

        context.state = AgentState.DECIDE

    def _serialize_result(self, result):

        data = {
            "is_error": result.is_error,
        }

        if result.structured_content is not None:

            data["structured_content"] = (
                result.structured_content
            )

        if result.content:

            content = []

            for block in result.content:

                if hasattr(block, "text"):

                    content.append(
                        block.text
                    )

                elif hasattr(block, "model_dump"):

                    content.append(
                        block.model_dump(
                            mode="json"
                        )
                    )

                else:

                    content.append(
                        str(block)
                    )

            data["content"] = content

        return json.dumps(
            data,
            ensure_ascii=False,
            default=str,
        )


