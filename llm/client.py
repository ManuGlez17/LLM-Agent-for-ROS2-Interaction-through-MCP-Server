from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient:

    def __init__(
        self,
        model="gpt-5.6",
    ):

        self.client = AsyncOpenAI()

        self.model = model

    async def generate(
        self,
        instructions,
        input_data,
        tools,
        previous_response_id=None,
    ):

        request = {
            "model": self.model,
            "instructions": instructions,
            "input": input_data,
            "tools": tools,
        }

        if previous_response_id:

            request[
                "previous_response_id"
            ] = previous_response_id

        return await self.client.responses.create(
            **request
        )