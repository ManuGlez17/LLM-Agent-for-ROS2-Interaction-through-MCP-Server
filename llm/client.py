from openai import AsyncOpenAI
from dotenv import load_dotenv

import base64
import io
from PIL import Image


def uploaded_image_to_data_url(uploaded_file):

    image = Image.open(uploaded_file)

    # Evita mandar imágenes enormes al modelo
    image.thumbnail((1280, 1280))

    # JPEG necesita RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=85,
    )

    image_bytes = buffer.getvalue()

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )

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
            "parallel_tool_calls": False,
        }

        if previous_response_id:

            request[
                "previous_response_id"
            ] = previous_response_id

        return await self.client.responses.create(
            **request
        )