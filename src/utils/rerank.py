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
        if model_type and model_type not in ['cross-encoder', 'flashrank', 't5', 'rankgpt', 'colbert']:
            raise ValueError("Unsupported model_type provided.")

        if model_type == 'cohere':
            return Reranker(rerank_model, lang=lang, api_key=api_key)
        elif model_type == 'jina':
            return Reranker(rerank_model, api_key=api_key)
        elif model_type == 'cross-encoder':
            return Reranker(rerank_model, model_type='cross-encoder')
        elif model_type == 'flashrank':
            return Reranker(rerank_model, model_type='flashrank')
        elif model_type == 't5':
            return Reranker(rerank_model, model_type='t5')
        elif model_type == 'rankgpt':
            return Reranker(rerank_model, model_type='rankgpt', api_key=api_key)
        elif model_type == 'colbert':
            return Reranker(rerank_model, model_type='colbert')
        else:
            return Reranker(model_name_or_path, model_type=model_type, api_key=api_key, api_provider=api_provider)
