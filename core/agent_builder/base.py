from core.utils import load_data
from abc import ABC, abstractmethod
from core.param_cache import ParamCache
from typing import Optional

class BaseRAGAgentBuilder(ABC):
    
    @property
    @abstractmethod
    def cache(self) -> ParamCache:
        """ Cache. """
    
class RAGAgentBuilder(BaseRAGAgentBuilder):
    
    def __init__(
        self, 
        cache: Optional[ParamCache] = None,
        ) -> None:
        
        self.cache = cache or ParamCache()
        
    @property
    def cache(self) -> ParamCache:
        return self.cache
    
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
        
        return "Data Loaded succesfully"