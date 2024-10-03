import base64
from pathlib import Path
import requests
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.chat_models import BaseChatModel

class LLMModel:
    """Base class for LLM models."""

    def __init__(self, model:BaseChatModel):
        self.model = model

    def _handle_request_error(self, e: requests.exceptions.RequestException) -> None:
        """Handle request exceptions."""
        print(f"API Request failed: {e}")

    def _handle_value_error(self, e: ValueError) -> None:
        """Handle value errors."""
        print(f"Value error: {e}")

    def runtask(self, param: str):
        raise NotImplemented



class LLMRAG(LLMModel):
    """Handles RAG tasks like recipe recommendations."""

    def __init__(self, model:BaseChatModel, prompt_template: str, vectors: VectorStore):
        super().__init__(model)
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.document_chain = create_stuff_documents_chain(self.model, self.prompt)
        self.retriever = vectors.as_retriever()
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)

    def runtask(self, user_message: str):
        try:
            response = self.retrieval_chain.invoke({'input': user_message})
            recipe_text = response.get("answer", "No recipe found.")
            print(f"Recipe found:\n{recipe_text}")
            return recipe_text

        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
        except ValueError as ve:
            self._handle_value_error(ve)



class LLMImage(LLMModel):
    """Handles image analysis tasks like text extraction."""

    def __init__(self, model:BaseChatModel, prompt: str):
        super().__init__(model)
        self.prompt= prompt

    def runtask(self, image_path: str):
        """Run Image task: Extract text from an image."""
        image_data = self._encode_image(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ],
        )

        try:
            response = self.model.invoke([message])
            extracted_text = response.content
            print(extracted_text)
            return extracted_text

        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
        except ValueError as ve:
            self._handle_value_error(ve)

    def _encode_image(self, image_path: str) -> str:
        """Encodes an image to base64 format."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
