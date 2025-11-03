"""
Мастер-модули VTuber AI системы.
Каждый мастер отвечает за свою доменную область.
"""

from .core_master import MasterCore
from .audio_master import MasterAudio
from .llm_master import MasterLLM
from .avatar_master import MasterAvatar

__all__ = ["MasterCore", "MasterAudio", "MasterLLM", "MasterAvatar"]