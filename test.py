from vanna.chromadb import ChromaDB_VectorStore
from vanna.elasticsearch import Elastic_VectorStore
from vanna.qianfan import (
  Qianfan_Chat,
  Qianfan_Embeddings,
  QianfanEmbeddingFunction,
)


class MyVanna(ChromaDB_VectorStore, Qianfan_Chat):
    def __init__(self, config=None):

        ChromaDB_VectorStore.__init__(self, config=config)
        Qianfan_Chat.__init__(self, config=config)


qianfan_embedding_function = QianfanEmbeddingFunction(config={ 'api_key': 'OOkmwXH5YjG2zhMzyxbpjGqp',
                     'secret_key':'0F1HM7G5Wj7WPuLEOGiLbHlXlyKFqlDB'})
    # config = {"api_key": "xxx", "secret_key": "xxx", "model": "erine-3.5","embedding_function":qianfan_embedding_function}


# embedding = Qianfan_Embeddings(config={'api_key': 'OOkmwXH5YjG2zhMzyxbpjGqp',
#                      'secret_key':'0F1HM7G5Wj7WPuLEOGiLbHlXlyKFqlDB'})
# vn = MyVanna(config={'bes_url': 'http://100.66.162.198:8020',
#                      'user': 'superuser',
#                      'password': '1234qwer_',
#                      "embedding_function":qianfan_embedding_function,
#                      'document_index': 'document_index',
#                      'ddl_index': 'ddl_index',
#                      'question_sql_index': 'question_sql_index',
#                      'api_key': 'OOkmwXH5YjG2zhMzyxbpjGqp',
#                      'secret_key':'0F1HM7G5Wj7WPuLEOGiLbHlXlyKFqlDB'})

vn = MyVanna(config={'path': './test',
                     "embedding_function": qianfan_embedding_function,
                     'document_index': 'document_index',
                     'ddl_index': 'ddl_index',
                     'question_sql_index': 'question_sql_index',
                     'api_key': 'OOkmwXH5YjG2zhMzyxbpjGqp',
                     'secret_key':'0F1HM7G5Wj7WPuLEOGiLbHlXlyKFqlDB'})


vn.add_documentation("电梯行业标准")

vn.generate_sql("物料表在哪个电梯?")
