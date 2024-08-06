
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from elasticsearch import Elasticsearch

from ..base import VannaBase


class Elastic_VectorStore(VannaBase):

  def __init__(self, config=None):
    VannaBase.__init__(self, config=config)
    if config is None:
      raise ValueError(
        "config is required, pass either a Pinecone client or an API key in the config."
      )

    client = config.get("client")
    bes_url = config.get("bes_url")
    user = config.get("user")
    passwd = config.get("password")

    self.document_index = config.get("document_index")
    self.document_index_type = config.get("document_index_type", "hnsw")
    self.document_index_query_field = config.get("document_index_query_field", "doc")
    self.document_index_vector_query_field = config.get("document_index_vector_query_field", "document")


    self.ddl_index = config.get("ddl_index")
    self.ddl_index_type = config.get("ddl_index_type", "hnsw")
    self.ddl_index_query_field = config.get("ddl_index_query_field", "doc")
    self.ddl_index_vector_query_field = config.get(
      "ddl_index_vector_query_field", "ddl")

    self.question_sql_index = config.get("question_sql_index")
    self.question_sql_index_type = config.get("question_sql_index_type", "hnsw")
    self.question_sql_index_query_field = config.get("question_sql_index_query_field", "doc")
    self.question_sql_index_vector_query_field = config.get(
      "question_sql_index_vector_query_field", "qa_sql")

    self.index_params = {}

    if config is not None and "index_params" in config:
      self.index_params = config.get("index_params")

    print("Elastic_VectorStore initialized with document_index: ",
          self.document_index, " ddl_index: ", self.ddl_index, " question_sql_index: ",
          self.question_sql_index)


    self.space_type = config.get("space_type", "cosine")

    if bes_url is not None:
      self.client = Elastic_VectorStore.bes_client(
        bes_url=bes_url, username=user, password=passwd,
      )
    else:
      raise ValueError("""Please specified a bes connection url.""")


    self.embedding = config.get("embedding")

    self.n_results = config.get("n_results", 10)
    self.fastembed_model = config.get("fastembed_model", "BAAI/bge-small-en-v1.5")

  def _create_index_if_not_exists(self, index_name, index_type, index_query_field, index_vector_field: str,
                                  dims_length: Optional[int] = None) -> None:
    """Create the index if it doesn't already exist.

    Args:
        dims_length: Length of the embedding vectors.
    """

    if self.client.indices.exists(index=index_name):
      print(
        f"Index {index_name} already exists. Skipping creation.")

    else:
      if dims_length is None:
        raise ValueError(
          "Cannot create index without specifying dims_length "
          + "when the index doesn't already exist. "
        )

      indexMapping = self._index_mapping(dims_length=dims_length, index_vector_field=index_vector_field, index_query_field=index_query_field, index_type=index_type)

      print(
        f"Creating index {index_name} with mappings {indexMapping}"
      )

      self.client.indices.create(
        index=index_name,
        body={
          "settings": {"index": {"knn": True}},
          "mappings": {"properties": indexMapping},
        },
      )


  def _bulk(self, index_name: str, index_type: str, index_query_field: str,  index_vector_field: str,
            texts: Optional[List[str]] = None,
            metadatas: Optional[List[Dict[Any, Any]]] = None,
             **kwargs) -> []:
    try:
      from elasticsearch.helpers import BulkIndexError, bulk
    except ImportError:
      raise ImportError(
        "Could not import elasticsearch python package. "
        "Please install it with `pip install elasticsearch`."
      )

    create_index_if_not_exists = kwargs.get("create_index_if_not_exists", True)
    refresh_indices = kwargs.get("refresh_indices", True)
    ids = kwargs.get("ids", [str(uuid.uuid4()) for _ in texts])
    requests = []

    if self.embedding is not None:
      embeddings = [self.embedding.generate_embedding(data=text) for text in texts]
      dims_length = len(embeddings[0])

      if create_index_if_not_exists:
        self._create_index_if_not_exists(dims_length=dims_length,
                                         index_type=index_type,
                                         index_query_field=index_query_field,
                                         index_vector_field=index_vector_field,
                                         index_name=index_name, )
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
          metadata = metadatas[i] if metadatas else {}

          requests.append(
            {
              "_op_type": "index",
              "_index": index_name,
              index_query_field: text,
              index_vector_field: vector,
              "metadata": metadata,
              "_id": ids[i],
            }
          )
    else:
      if create_index_if_not_exists:
        self._create_index_if_not_exists(index_query_field=index_query_field,
                                         index_type=index_type,
                                         index_name=index_name, )
        for i, text in enumerate(texts):
          metadata = metadatas[i] if metadatas else {}

          requests.append(
            {
              "_op_type": "index",
              "_index": index_name,
              index_query_field: text,
              "metadata": metadata,
              "_id": ids[i],
            }
          )

    if len(requests) > 0:
      try:
        success, failed = bulk(
          self.client, requests, stats_only=True, refresh=refresh_indices
        )
        print(
          f"Added {success} and failed to add {failed} texts to index"
        )
        return ids
      except BulkIndexError as e:
        print(f"Error adding texts: {e}")
        firstError = e.errors[0].get("index", {}).get("error", {})
        print(f"First error reason: {firstError.get('reason')}")
        raise e

    else:
      print("No texts to add to index")
      return []


  def get_similar_question_sql(self, question: str, **kwargs) -> list:
    pass

  def get_related_ddl(self, question: str, **kwargs) -> list:
    pass

  def get_related_documentation(self, question: str, **kwargs) -> list:
    pass

  def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
    ids = self._bulk(index_name=self.question_sql_index,
                     index_type=self.question_sql_index_type,
                     index_query_field=self.document_index_query_field,
                     index_vector_field=self.document_index_vector_query_field,
                     texts=[question],
                     metadatas=[{"sql": sql, "question": question}])

    if ids is not None and len(ids) > 0:
      return ids[0]
    return ""

  def add_ddl(self, ddl: str, **kwargs) -> str:
    ids = self._bulk(index_name=self.ddl_index,
                     index_type=self.ddl_index_type,
                     index_query_field=self.ddl_index_query_field,
                     index_vector_field=self.ddl_index_vector_query_field,
                     texts=[ddl], metadatas=[{"ddl": ddl}])

    if ids is not None and len(ids) > 0:
      return ids[0]
    return ""


  def add_documentation(self, documentation: str, **kwargs) -> str:
    ids = self._bulk(index_name=self.document_index,
                     index_type=self.document_index_type,
                     index_query_field=self.document_index_query_field,
                     index_vector_field=self.document_index_vector_query_field,
                     texts=[documentation], metadatas=[{"documentation": documentation}])

    if ids is not None and len(ids) > 0:
      return ids[0]
    return ""


  def _search(self,
              index_name: str,
              index_vector_field: str,
              index_query_field: str,
              index_type: str,
              query: str,
              query_vector: Union[List[float], None],
              custom_query: Optional[Callable[[Dict, Union[str, None]], Dict]] = None,
              filter: Optional[dict] = None,
              search_params: Dict = {},
              **kwargs) -> List[str]:


    if self.embedding and query is not None:
      query_vector = self.embedding.generate_embedding(data=query)

    query_body = self._query_body(
      vector_index_field=index_vector_field,
      index_type=index_type,
      query_vector=query_vector, filter=filter, search_params=search_params
    )

    if custom_query is not None:
      query_body = custom_query(query_body, query)
      print(f"Calling custom_query, Query body now: {query_body}")

    print(f"Query body: {query_body}")

    # Perform the kNN search on the BES index and return the results.
    response = self.client.search(index=index_name, body=query_body)
    print(f"response={response}")

    return [hit['_source'][index_query_field] for hit in response['hits']['hits']]

  def _query_body(
    self,
    vector_index_field: str,
    index_type: str,
    query_vector: Union[List[float], None],
    filter: Optional[dict] = None,
    search_params: Dict = {},
  ) -> Dict:
    query_vector_body = {"vector": query_vector,
                         "k": search_params.get("k", 2)}

    if filter is not None and len(filter) != 0:
      query_vector_body["filter"] = filter

    if "linear" == index_type:
      query_vector_body["linear"] = True
    else:
      query_vector_body["ef"] = search_params.get("ef", 10)

    return {
      "size": search_params.get("size", self.n_results),
      "query": {"knn": {vector_index_field: query_vector_body}},
    }

  def _index_mapping(self, dims_length: Union[int, None], index_vector_field, index_query_field: str, index_type: str) -> Dict:
    """
    Executes when the index is created.

    Args:
        dims_length: Numeric length of the embedding vectors,
                    or None if not using vector-based query.
        index_params: The extra pamameters for creating index.

    Returns:
        Dict: The Elasticsearch settings and mappings for the strategy.
    """
    if "linear" == index_type:
      return {
        index_vector_field: {
          "type": "bpack_vector",
          "dims": dims_length,
          "build_index": self.index_params.get("build_index", False),
        },
        index_query_field: {
          "type": "text"
        }
      }

    elif "hnsw" == index_type:
      return {
        index_vector_field: {
          "type": "bpack_vector",
          "dims": dims_length,
          "index_type": "hnsw",
          "space_type": self.space_type,
          "parameters": {
            "ef_construction": self.index_params.get(
              "hnsw_ef_construction", 200
            ),
            "m": self.index_params.get("hnsw_m", 4),
          },
        },
        index_query_field: {
          "type": "text"
        }
      }
    else:
      return {
        index_vector_field: {
          "type": "bpack_vector",
          "model_id": self.index_params.get("model_id", ""),
        },
        index_query_field: {
          "type": "text"
        }
      }


  @staticmethod
  def bes_client(
    *,
    bes_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
  ) -> "Elasticsearch":
    try:
      import elasticsearch
    except ImportError:
      raise ImportError(
        "Could not import elasticsearch python package. "
        "Please install it with `pip install elasticsearch`."
      )

    connection_params: Dict[str, Any] = {}

    connection_params["hosts"] = [bes_url]
    if username and password:
      connection_params["basic_auth"] = (username, password)

    es_client = elasticsearch.Elasticsearch(**connection_params)
    try:
      es_client.info()
    except Exception as e:
      print(f"Error connecting to Elasticsearch: {e}")
      raise e
    return es_client
