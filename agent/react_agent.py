import json

from .states import AgentState
from .context import AgentContext

import time


class ReActAgent:

    def __init__(
        self,
        llm,
        mcp_client,
        tools,
        logger=None,
        model="gpt-5.6-terra",
    ):

        self.llm = llm
        self.mcp = mcp_client

        self.tools = tools
        self.model = model

        self.logger = logger

        self.max_steps = 100
        self.previous_response_id = None


    def _log(
        self,
        event,
        data=None,
    ):

        if self.logger:

            self.logger.log_agent(
                event,
                data,
            )

    def _log_experiment(
        self,
        event,
        data=None,
    ):

        if self.logger:

            self.logger.log_experiment(
                event,
                data,
            )


    async def run(self, goal: str, image_url = None):

        task_start = time.perf_counter()

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

            message = (
                f"\n[{context.step}] "
                f"STATE = {context.state.name}"
            )

            print(message)

            self._log("step", {"step": context.step, "state": context.state.name})

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

        context.task_time += time.perf_counter() - task_start

        self._log_experiment(
                    f"episode_{self.logger.episode_number}",
                    {
                        "goal": context.goal,
                        "step": context.step,
                        "final_answer":
                            context.final_answer,
                        "task_time":
                            context.task_time,
                        "llm_calls":
                            context.llm_calls,
                        "tool_calls":
                            context.tool_calls,
                        "llm_time":
                            context.llm_time,
                        "tool_time":
                            context.tool_time,
                        "input_tokens":
                            context.input_tokens,
                        "output_tokens":
                            context.output_tokens,
                    },
                )
        
        self.logger.episode_number += 1

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

            llm_start = time.perf_counter()

            response = await self.llm.generate(
                    instructions=instructions,
                    input_data=input_data,
                    tools=self.tools,
                    previous_response_id=(
                        context.previous_response_id
                    ),
                )
            
            context.llm_time += time.perf_counter() - llm_start
            context.llm_calls += 1

            context.input_tokens += (
                            response.usage.input_tokens
                        )
            
            context.output_tokens += (
                            response.usage.output_tokens
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

            llm_start = time.perf_counter()

            response = await self.llm.generate(
                instructions=instructions,
                input_data=input_data,
                tools=self.tools,
                previous_response_id=(
                    context.previous_response_id
                ),
            )

            context.llm_time += time.perf_counter() - llm_start
            context.llm_calls += 1

            context.input_tokens += (
                response.usage.input_tokens
            )

            context.output_tokens += (
                response.usage.output_tokens
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

            print("Accion seleccionada:", context.pending_call.name)

            self._log(
                "TOOL_SELECTED",
                {
                    "step": context.step,
                    "tool":
                        context.pending_call.name,

                    "arguments":
                        context.pending_call.arguments,

                    "call_id":
                        context.pending_call.call_id,
                },
            )
            context.state = AgentState.ACT

            return

        context.final_answer = response.output_text

        context.state = AgentState.DONE

        self._log(
            "DONE",
            {
                "step": context.step,
                "answer":
                    context.final_answer,
            },
        )

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

            
            self._log(
                            "ACT",
                            {
                                "step": context.step,
                                "tool": call.name,
                                "arguments": arguments,
                            },
            )

            tool_start = time.perf_counter()
        
            result = await self.mcp.call_tool(
                call.name,
                arguments,
            )

            context.tool_time += time.perf_counter() - tool_start
            context.tool_calls += 1

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

            if "base64" in context.observation:
                logged_result = "[image data omitted]"
            else:
                logged_result = context.observation
            self._log(
                "MCP_RESULT",
                {
                    "step": context.step,
                    "tool": call.name,
                    "result": logged_result,
                },
            )

            context.state = AgentState.OBSERVE

        except Exception as error:

            context.observation = json.dumps({
                "error": str(error)
            })

            self._log(
                "ERROR",
                {
                    "step": context.step,
                    "tool": call.name,
                    "error": str(error),
                },
            )

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


