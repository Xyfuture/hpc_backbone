from enum import Enum
from pydantic import BaseModel


class HDStrategy(str, Enum):
    ORIGINAL = "Original"
    RESIZE = "Resize"
    CROP = "Crop"


class Config(BaseModel):

    hd_strategy: str
    hd_strategy_crop_margin: int
    hd_strategy_crop_trigger_size: int
    hd_strategy_resize_limit: int

    # prompt: str = ""
    # negative_prompt: str = ""
    # # 始终是在原图尺度上的值
    # use_croper: bool = False
    # croper_x: int = None
    # croper_y: int = None
    # croper_height: int = None
    # croper_width: int = None
