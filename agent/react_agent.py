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
        self.previous_response_id = None

    async def run(self, goal: str, image_url = None):

        context = AgentContext(
            goal=goal,
            previous_response_id=self.previous_response_id,
        )

        context.image_url = image_url
        
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

        self.previous_response_id = (
            context.previous_response_id
        )

        return context.final_answer

    async def _decide(self, context):
        instructions = """
    Eres un agente robótico que utiliza herramientas MCP
    para interactuar con ROS.

    Usa observaciones reales del sistema.
    No inventes resultados.
    Si el usuario proporciona una imagen adjunta directamente
    como entrada multimodal, analiza esa imagen directamente.

    NO utilices view_saved_image para analizar una imagen adjunta
    por el usuario.
    """

        # ====================================
        # CASO 1:
        # Nuevo mensaje del usuario
        # ====================================

        if context.observation is None:

            if context.image_url is not None:

                input_data = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": context.goal,
                            },
                            {
                                "type": "input_image",
                                "image_url": context.image_url,
                                "detail": "high",
                            },
                        ],
                    }
                ]

            else:
                input_data = context.goal

            response = await self.llm.generate(
                    instructions=instructions,
                    input_data=input_data,
                    tools=self.tools,
                    previous_response_id=(
                        context.previous_response_id
                    ),
                )

        # ====================================
        # CASO 2:
        # Resultado de una herramienta MCP
        # ====================================

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
                input_data=input_data,
                tools=self.tools,
                previous_response_id=(
                    context.previous_response_id
                ),
            )
            context.observation = None
        context.previous_response_id = response.id

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if tool_calls:

            context.pending_call = tool_calls[0]

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


