# Standard Library Imports
import sys
from typing import Union

# Third-Party Imports
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Custom Imports
from src.utils.logger import logging
from src.utils.exception import CustomException


class EmbeddingManager:

    def embed_query(self, text: str) -> list:
        """
        LangChain compatibility: Embed a query string for vector search.
        Returns a list for Pinecone serialization.
        """
        return self.embed_text(text).tolist()
    """
    Multi-Modal Embedding Manager using CLIP.

    Responsibilities:
        - Generate text embeddings
        - Generate image embeddings
        - Support hybrid retrieval pipelines
        - Provide normalized embeddings for Pinecone storage

    Model:
        openai/clip-vit-base-patch32

    Supported Modalities:
        - Text
        - Images
        - Charts
        - Figures
    """

    # Singleton model instances
    _clip_model = None
    _clip_processor = None
    _device = None

    def __init__(self) -> None:
        """
        Initialize CLIP model and processor.

        Uses lazy singleton initialization to avoid
        repeatedly loading large models into memory.
        """

        try:
            logging.info(
                "Initializing EmbeddingManager."
            )

            # ---------------------------------------------------
            # Device Configuration
            # ---------------------------------------------------
            if EmbeddingManager._device is None:

                EmbeddingManager._device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )

                logging.info(
                    f"Using device: "
                    f"{EmbeddingManager._device}"
                )

            # ---------------------------------------------------
            # Lazy Load CLIP Model
            # ---------------------------------------------------
            if EmbeddingManager._clip_model is None:

                logging.info(
                    "Loading CLIP model..."
                )

                EmbeddingManager._clip_model = (
                    CLIPModel.from_pretrained(
                        "openai/clip-vit-base-patch32"
                    ).to(EmbeddingManager._device)
                )

                logging.info(
                    "CLIP model loaded successfully."
                )

            # ---------------------------------------------------
            # Lazy Load CLIP Processor
            # ---------------------------------------------------
            if EmbeddingManager._clip_processor is None:

                logging.info(
                    "Loading CLIP processor..."
                )

                EmbeddingManager._clip_processor = (
                    CLIPProcessor.from_pretrained(
                        "openai/clip-vit-base-patch32"
                    )
                )

                logging.info(
                    "CLIP processor loaded successfully."
                )

        except Exception as e:
            logging.exception(
                "Failed to initialize EmbeddingManager."
            )
            raise CustomException(e, sys)

    @classmethod
    def embed_text(
        cls,
        text: str,
    ) -> np.ndarray:
        """
        Generate CLIP text embedding.

        Args:
            text (str):
                Input text chunk.

        Returns:
            np.ndarray:
                Normalized text embedding vector.
        """

        try:
            # Ensure model is initialized
            cls()

            logging.info(
                "Generating text embedding."
            )

            inputs = cls._clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,  # CLIP token limit
            )

            # Move tensors to device
            inputs = {
                key: value.to(cls._device)
                for key, value in inputs.items()
            }

            with torch.no_grad():

                text_features = cls._clip_model.get_text_features(
                    **inputs
                )
                # If output is not a tensor, get the tensor (for compatibility)
                if not isinstance(text_features, torch.Tensor):
                    # Try common attributes
                    if hasattr(text_features, "pooler_output"):
                        text_features = text_features.pooler_output
                    elif hasattr(text_features, "last_hidden_state"):
                        text_features = text_features.last_hidden_state
                    else:
                        raise CustomException("Unknown output type from get_text_features", sys)

                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            return (
                text_features
                .squeeze()
                .cpu()
                .numpy()
            )

        except Exception as e:
            logging.exception(
                "Error while generating text embedding."
            )
            raise CustomException(e, sys)

    @classmethod
    def embed_image(
        cls,
        image_data: Union[str, Image.Image],
    ) -> np.ndarray:
        """
        Generate CLIP image embedding.

        Args:
            image_data:
                Either:
                    - Image file path
                    - PIL Image object

        Returns:
            np.ndarray:
                Normalized image embedding vector.
        """

        try:
            # Ensure model is initialized
            cls()

            logging.info(
                "Generating image embedding."
            )

            # ---------------------------------------------------
            # Load Image
            # ---------------------------------------------------
            if isinstance(image_data, str):

                image = Image.open(
                    image_data
                ).convert("RGB")

            else:

                image = image_data.convert("RGB")

            # ---------------------------------------------------
            # Preprocess Image
            # ---------------------------------------------------
            inputs = cls._clip_processor(
                images=image,
                return_tensors="pt",
            )

            # Move tensors to device
            inputs = {
                key: value.to(cls._device)
                for key, value in inputs.items()
            }

            # ---------------------------------------------------
            # Generate Embedding
            # ---------------------------------------------------
            with torch.no_grad():

                image_features = (
                    cls._clip_model.get_image_features(
                        **inputs
                    )
                )

                # Normalize embeddings
                image_features = (
                    image_features
                    / image_features.norm(
                        dim=-1,
                        keepdim=True,
                    )
                )

            return (
                image_features
                .squeeze()
                .cpu()
                .numpy()
            )

        except Exception as e:
            logging.exception(
                "Error while generating image embedding."
            )
            raise CustomException(e, sys)