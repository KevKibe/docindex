from rerankers import Reranker

class RerankerConfig:
    @staticmethod
    def get_ranker(rerank_model: str, model_type: str = None, lang: str = None, api_key: str = None, api_provider: str = None) -> Reranker:
        """
        Returns a Reranker instance based on the provided parameters.

        Args:
            rerank_model (str): The name or path of the model.
            model_type (str, optional): The type of the model. Defaults to None.
            lang (str, optional): The language for multilingual models. Defaults to None.
            api_key (str, optional): The API key for models accessed through an API. Defaults to None.
            api_provider (str, optional): The provider of the API. Defaults to None.

        Returns:
            Reranker: An instance of Reranker.

        Raises:
            ValueError: If unsupported model_type is provided.
        """
        if rerank_model and rerank_model not in ["cross-encoder", "flashrank", "t5", "rankgpt", "colbert", "mixedbread-ai/mxbai-rerank-large-v1", "ce-esci-MiniLM-L12-v2", "unicamp-dl/InRanker-base", "jina",
                                             "rankgpt", "rankgpt3"]:
            raise ValueError("Unsupported model_type provided.")

        if rerank_model == 'cohere':
            return Reranker(rerank_model, lang=lang, api_key=api_key)
        elif rerank_model == 'jina':
            return Reranker(rerank_model, api_key=api_key)
        elif rerank_model == 'cross-encoder':
            return Reranker(rerank_model)
        elif rerank_model == 'flashrank':
            return Reranker(rerank_model)
        elif rerank_model == 't5':
            return Reranker(rerank_model)
        elif rerank_model == 'rankgpt':
            return Reranker(rerank_model, api_key=api_key)
        elif rerank_model == 'rankgpt3':
            return Reranker(rerank_model, api_key=api_key)
        elif rerank_model == 'colbert':
            return Reranker(rerank_model)
        elif rerank_model == "mixedbread-ai/mxbai-rerank-large-v1":
            return Reranker(rerank_model, model_type='cross-encoder')
        elif rerank_model == "ce-esci-MiniLM-L12-v2":
            return Reranker(rerank_model, model_type='flashrank')
        elif rerank_model == "unicamp-dl/InRanker-base":
            return Reranker(rerank_model, model_type='t5')
        else:
            return None
