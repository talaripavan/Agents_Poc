# from core.utils import load_data
from core.utils import load_data
from pydantic import BaseModel , Field
from typing import Optional
from pathlib import Path
import json

save_dir = "cache"

class ParamCache(BaseModel):
    
    class Config:
        arbitrary_types_allowed = True
        
    file_names: list[str] = Field(
        default_factory=list, description="File names as data source (if specified)"
    )
    directory : Optional[str] = Field(
        default=None, description="File names as data source (if specified)"
    )   
    docs : list = Field(default_factory=list,description="Documents for Rag Agent")
    
    def save_to_disk(self , save_dir :str) -> None:
        
        dict_to_serialize = {
            "file_names" : self.file_names,
            "directory" : self.directory,
        }
        
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)
        with open(Path(save_dir)/ "cache.json", "w") as f:
            json.dump(dict_to_serialize,f)
  
    @classmethod
    def load_from_disk(
        cls,
        save_dir: str,
        ) -> "ParamCache":
        
        with open(Path(save_dir)/ "cache.json","r") as f:
            cache_dict = json.load(f)
            
        cache_dict["docs"] = load_data(
            file_names=cache_dict["file_names"],
            directory= cache_dict["directory"],
            )
        return cls(**cache_dict)
'''
cache = ParamCache()
cache.save_to_disk(save_dir)
'''


