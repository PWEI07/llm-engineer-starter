from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from src.reranker import BgeRerank


class RAG(object):
    def __init__(self, vectordb, k=10,
                 llm_path=r'/Users/peter_zirui_wei/PycharmProjects/llama.cpp/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
                 embedding_model='BAAI/bge-large-en-v1.5', *args, **kwargs):
        self.retriever = vectordb.as_retriever(search_kwargs={"k": k})
        self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)

        self.llm = LlamaCpp(
            model_path=llm_path,
            temperature=0,
            max_tokens=200,
            n_gpu_layers=-1,
            top_p=1,
            echo=False,
            f16_kv=True,
            verbose=True,
        )
        self.redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embedding_function)
        self.reranker = BgeRerank()
        pipeline_compressor = DocumentCompressorPipeline(transformers=[self.redundant_filter,
                                                                       self.reranker])
        compression_pipeline = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                              base_retriever=self.retriever)
        self.rag = RetrievalQA.from_chain_type(llm=self.llm,
                                               chain_type="stuff",
                                               retriever=compression_pipeline,
                                               return_source_documents=True)

    def query(self, query):
        result = self.rag({"query": query})
        return result
