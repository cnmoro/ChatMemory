from minivectordb.embedding_model import EmbeddingModel, AlternativeModel as EmbeddingModelType
from enum import Enum

model = None

class AlternativeModel(str, Enum):
    tiny  = "tiny"
    small = "small"
    large = "large"
    bgem3 = "bgem3"

def extract_embeddings_free(text, _type: AlternativeModel, reload_model=False):
    global model

    if reload_model:
        model = None

    if model is None:
        if _type == AlternativeModel.tiny:
            model = EmbeddingModel(use_quantized_onnx_model=True)
        else:
            _type_map = {
                AlternativeModel.small: EmbeddingModelType.small,
                AlternativeModel.large: EmbeddingModelType.large,
                AlternativeModel.bgem3: EmbeddingModelType.bgem3
            }
            model = EmbeddingModel(
                use_quantized_onnx_model=False,
                alternative_model=_type_map[_type]
            )

    return model.extract_embeddings(text)

def extract_embeddings_openai(text, openai_client, model_name = "text-embedding-3-small"):
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(input = [text], model=model_name).data[0].embedding
