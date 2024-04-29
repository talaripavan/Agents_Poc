from core.utils import load_data
from abc import ABC, abstractmethod
from core.param_cache import ParamCache
from typing import Optional
from core.constants import AGENT_CACHE_DIR

save_dir = "cache"

class BaseRAGAgentBuilder(ABC):
    
    @property
    @abstractmethod
    def cache(self) -> ParamCache:
        """ Cache. """

class RAGAgentBuilder(BaseRAGAgentBuilder):
    
    def __init__(self, cache: Optional[ParamCache] = None,) -> None:
        self._cache = cache or ParamCache()
        
    @property
    def cache(self) -> ParamCache:
        """Cache."""
        return self._cache

    def load_data(
        self,
        file_names:Optional[list[str]] = None,
        directory: Optional[list[str]] = None,
        ) -> str:
        file_names = file_names or []
        directory = directory or ""
        
        docs = load_data(file_names=file_names,directory=directory)
        self.cache.docs = docs
        self.cache.file_names = file_names
        self.cache.directory = directory
        
        #return "Data Loaded succesfully"
        return docs



