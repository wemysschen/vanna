import qianfan
from chromadb import Documents, EmbeddingFunction, Embeddings

from ..base import VannaBase


class Qianfan_Embeddings(VannaBase):
  def __init__(self, client=None, config=None):
    VannaBase.__init__(self, config=config)

    if client is not None:
      self.client = client
      return

    if "api_key" not in config:
      raise Exception("Missing api_key in config")
    self.api_key = config["api_key"]

    if "secret_key" not in config:
      raise Exception("Missing secret_key in config")
    self.secret_key = config["secret_key"]

    self.client = qianfan.Embedding(ak=self.api_key, sk=self.secret_key)

  def generate_embedding(self, data: str, **kwargs) -> list[float]:
    if self.config is not None and "model" in self.config:
      embedding = self.client.do(
        model=self.config["model"],
        input=[data],
      )
    else:
      embedding = self.client.do(
        model="bge-large-zh",
        input=[data],
      )

    return embedding.get("data")[0]["embedding"]

class QianfanEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A embeddingFunction that uses ZhipuAI to generate embeddings which can use in chromadb.
    usage:
    class MyVanna(ChromaDB_VectorStore, Qianfan_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            Qianfan_Chat.__init__(self, config=config)

    config={'api_key': 'xxx'}
    qianfan_embedding_function = QianfanEmbeddingFunction(config=config)
    config = {"api_key": "xxx", "secret_key": "xxx", "model": "erine-3.5","embedding_function":qianfan_embedding_function}

    vn = MyVanna(config)

    """

    def __init__(self, config=None):
      if config is None or "api_key" not in config:
        raise ValueError("Missing 'api_key' in config")

      if "api_key" not in config:
        raise Exception("Missing api_key in config")
      self.api_key = config["api_key"]

      if "secret_key" not in config:
        raise Exception("Missing secret_key in config")
      self.secret_key = config["secret_key"]

      try:
        self.client = qianfan.Embedding(ak=self.api_key, sk=self.secret_key)
      except Exception as e:
        raise ValueError(f"Error initializing Qianfan embedding client: {e}")

      self.model_name = config.get("model", "bge-large-zh")

    def __call__(self, input: Documents) -> Embeddings:
      # Replace newlines, which can negatively affect performance.
      input = [t.replace("\n", " ") for t in input]
      all_embeddings = []
      print(f"Generating embeddings for {len(input)} documents")

      # Iterating over each document for individual API calls
      try:
        response = self.client.do(
          model="bge-large-zh",
          texts=input,
        )
        # print(response)
        all_embeddings = [data["embedding"] for data in response.get("data")]
      except Exception as e:
        raise ValueError(f"Error generating embedding for document: {e}")
      return all_embeddings
