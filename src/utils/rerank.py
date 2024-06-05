from rerankers import Reranker

class RerankerConfig:
    SUPPORTED_MODELS = {
        'cohere': {'lang': True, 'api_key': True},
        'jina': {'api_key': True},
        'cross-encoder': {},
        'flashrank': {},
        't5': {},
        'rankgpt': {'api_key': True},
        'rankgpt3': {'api_key': True},
        'colbert': {},
        'mixedbread-ai/mxbai-rerank-large-v1': {'model_type': True},
        'ce-esci-MiniLM-L12-v2': {'model_type': True},
        'unicamp-dl/InRanker-base': {'model_type': True},
    }
    @staticmethod
    def get_ranker(rerank_model: str, lang: str = None, api_key: str = None, model_type: str = None) -> Reranker:
        """
        Returns a Reranker instance based on the provided parameters.

        Args:
            rerank_model (str): The name or path of the model.
            lang (str, optional): The language for multilingual models. Defaults to None.
            api_key (str, optional): The API key for models accessed through an API. Defaults to None.
            model_type (str, optional): The model type of a reranker, defaults to None.

        Returns:
            Reranker: An instance of Reranker.

        Raises:
            ValueError: If unsupported rerank_model is provided.
        """
        if rerank_model not in RerankerConfig.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported rerank_model provided: {rerank_model}")

        model_config = RerankerConfig.SUPPORTED_MODELS[rerank_model]
        init_kwargs = {
            'lang': lang if model_config.get('lang') else None,
            'api_key': api_key if model_config.get('api_key') else None,
            'model_type': model_type if model_config.get('model_type') else None
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        return Reranker(rerank_model, **init_kwargs)

