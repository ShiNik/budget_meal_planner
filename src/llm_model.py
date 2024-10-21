import base64
from pathlib import Path

import requests
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from logger import get_logger

recipes_logger = get_logger("recipes")


class LLMModel:
    """Base class for LLM models."""

    def __init__(self, model: BaseChatModel):
        self.model = model

    def _handle_request_error(self, e: requests.exceptions.RequestException) -> None:
        """Handle request exceptions."""
        recipes_logger.info(f"API Request failed: {e}")

    def _handle_value_error(self, e: ValueError) -> None:
        """Handle value errors."""
        recipes_logger.info(f"Value error: {e}")

    async def runtask(self, param: str):
        raise NotImplementedError


class LLMRAG(LLMModel):
    """Handles RAG tasks like recipe recommendations."""

    def __init__(
        self,
        model: BaseChatModel,
        prompt_template: str,
        vectors: VectorStore,
    ):
        super().__init__(model)
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.document_chain = create_stuff_documents_chain(self.model, self.prompt)
        self.retriever = vectors.as_retriever(search_type="mmr", search_kwargs={"k": 1})
        self.retrieval_chain = create_retrieval_chain(
            self.retriever,
            self.document_chain,
        )

    async def runtask(self, user_message: str) -> str:
        try:
            response = await self.retrieval_chain.ainvoke({"input": user_message})
            recipe_text = response.get("answer", "No recipe found.")
            recipes_logger.info(f"Recipe found:\n{recipe_text}")
            return recipe_text

        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
        except ValueError as ve:
            self._handle_value_error(ve)


class LLMImage(LLMModel):
    """Handles image analysis tasks like text extraction."""

    def __init__(self, model: BaseChatModel, prompt: str):
        super().__init__(model)
        self.prompt = prompt

    async def runtask(self, image_path: str) -> str:
        """Run Image task: Extract text from an image."""
        image_data = self._encode_image(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": self.prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )

        try:
            response = await self.model.ainvoke([message])
            return response.content

        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
        except ValueError as ve:
            self._handle_value_error(ve)

    def _encode_image(self, image_path: str) -> str:
        with Path(image_path).open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
