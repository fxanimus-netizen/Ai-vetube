"""
osc.py — финальная версия
- Восстановлен Luppet MIDI Pose Bridge (без головы: Head исключён из MIDI).
- Асинхронная set_emotion() (совместима с `await avatar.set_emotion(...)`).
- Поддержка VSeeFace/Unity по VMC: /VMC/Ext/Blend/Apply, /VMC/Ext/Bone/Pos, /VMC/Ext/Talk.
- Неблокирующая pulse_emotion_async().
"""

from __future__ import annotations

import time
import logging
import asyncio
from typing import Dict, Tuple, Optional

from pythonosc.udp_client import SimpleUDPClient

# =========================
# MIDI (для Luppet)
# =========================
try:
    import mido
    _MIDI_AVAILABLE = True
except Exception:
    mido = None
    _MIDI_AVAILABLE = False

logger = logging.getLogger("AvatarOSC")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# =========================
# LUPPET MIDI POSE BRIDGE (без головы)
# =========================
class LuppetPoseBridge:
    """
    Преобразует позы (кости) в MIDI CC для Luppet.
    ГОЛОВА ИСКЛЮЧЕНА: управляется VSeeFace (через VMC).
    Значения CC: 0..127 (нормализация осей -1..+1 → 0..127).
    """

    def __init__(self, port_name: str = "LuppetBridge", virtual: bool = True):
        if not _MIDI_AVAILABLE:
            raise RuntimeError("mido не установлен — MIDI-бридж недоступен")
        try:
            if virtual:
                self.out = mido.open_output(port_name, virtual=True)
                logger.info(f"🎛️ MIDI виртуальный порт создан: {port_name}")
            else:
                self.out = mido.open_output(port_name)
                logger.info(f"🎛️ MIDI порт открыт: {self.out.name}")
        except Exception as e:
            # Фоллбек: открыть любой доступный порт
            logger.warning(f"Не удалось открыть MIDI-порт '{port_name}': {e}")
            self.out = mido.open_output()
            logger.info(f"🎛️ Используется существующий MIDI-порт: {getattr(self.out, 'name', 'unknown')}")

        # CC-мэппинг ТОЛЬКО ДЛЯ ТЕЛА (без Head)
        # Axes: 'x','y','z'
        self.cc_map: Dict[Tuple[str, str], int] = {
            # Туловище
            ("Spine", "x"): 20, ("Spine", "y"): 21, ("Spine", "z"): 22,
            ("Hips", "x"): 70, ("Hips", "y"): 71, ("Hips", "z"): 72,
            # Левая рука
            ("LeftUpperArm", "x"): 30, ("LeftUpperArm", "y"): 31, ("LeftUpperArm", "z"): 32,
            ("LeftLowerArm", "x"): 33, ("LeftLowerArm", "y"): 34, ("LeftLowerArm", "z"): 35,
            ("LeftHand", "x"): 36, ("LeftHand", "y"): 37, ("LeftHand", "z"): 38,
            # Правая рука
            ("RightUpperArm", "x"): 40, ("RightUpperArm", "y"): 41, ("RightUpperArm", "z"): 42,
            ("RightLowerArm", "x"): 43, ("RightLowerArm", "y"): 44, ("RightLowerArm", "z"): 45,
            ("RightHand", "x"): 46, ("RightHand", "y"): 47, ("RightHand", "z"): 48,
            # Левая нога
            ("LeftUpperLeg", "x"): 50, ("LeftUpperLeg", "y"): 51, ("LeftUpperLeg", "z"): 52,
            ("LeftLowerLeg", "x"): 53, ("LeftLowerLeg", "y"): 54, ("LeftLowerLeg", "z"): 55,
            ("LeftFoot", "x"): 56, ("LeftFoot", "y"): 57, ("LeftFoot", "z"): 58,
            # Правая нога
            ("RightUpperLeg", "x"): 60, ("RightUpperLeg", "y"): 61, ("RightUpperLeg", "z"): 62,
            ("RightLowerLeg", "x"): 63, ("RightLowerLeg", "y"): 64, ("RightLowerLeg", "z"): 65,
            ("RightFoot", "x"): 66, ("RightFoot", "y"): 67, ("RightFoot", "z"): 68,
        }

    @staticmethod
    def _to_cc(v: float) -> int:
        if v is None:
            v = 0.0
        v = max(-1.0, min(1.0, float(v)))
        return int(round((v + 1.0) * 63.5))

    def send_pose(self, bone: str, position, rotation) -> None:
        """Преобразует rotation=(rx,ry,rz) в MIDI CC и отправляет в Luppet."""
        try:
            rx, ry, rz = (rotation[0], rotation[1], rotation[2])
        except Exception:
            return
        for axis, val in zip(("x", "y", "z"), (rx, ry, rz)):
            cc = self.cc_map.get((bone, axis))
            if cc is None:
                continue
            value = self._to_cc(val)
            try:
                msg = mido.Message("control_change", control=cc, value=value)
                self.out.send(msg)
            except Exception:
                pass

    def close(self) -> None:
        try:
            self.out.close()
        except Exception:
            pass


class MultiTargetOSCController:
    """
    Контроллер OSC для Luppet, VSeeFace и Unity.
    - Эмоции/мимика → VSeeFace/Unity (и Luppet в формате BlendShape).
    - Позы/кости → VSeeFace/Unity по VMC, параллельно → Luppet через MIDI (без головы).
    - Асинхронная set_emotion() для безопасного await-вызова.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        luppet_port: int = 39539,
        vseeface_port: int = 39540,
        unity_port: int = 39541,
        enable_unity: bool = True,
        enable_luppet_midi: bool = True,
        luppet_midi_port_name: str = "LuppetBridge",
        luppet_midi_virtual: bool = True,
    ):
        self.host = host
        self.enable_unity = enable_unity
        try:
            self.luppet = SimpleUDPClient(host, luppet_port)
            self.vseeface = SimpleUDPClient(host, vseeface_port)
            self.unity = SimpleUDPClient(host, unity_port)
            logger.info(
                f"OSC → Luppet:{luppet_port} | VSeeFace:{vseeface_port} | Unity:{unity_port}"
            )
        except Exception as e:
            logger.error(f"OSC init error: {e}")

        # MIDI-мост к Luppet
        self.luppet_bridge: Optional[LuppetPoseBridge] = None
        if enable_luppet_midi and _MIDI_AVAILABLE:
            try:
                self.luppet_bridge = LuppetPoseBridge(
                    port_name=luppet_midi_port_name,
                    virtual=luppet_midi_virtual,
                )
            except Exception as e:
                logger.warning(f"LuppetPoseBridge disabled: {e}")
        elif enable_luppet_midi and not _MIDI_AVAILABLE:
            logger.warning("mido/rtmidi не найдены — LuppetPoseBridge неактивен")

    # ============================== ЭМОЦИИ ==============================

    def send_emotion(self, emotion: str, value: float = 1.0):
        """Отправка эмоции (BlendShape) во все клиенты"""
        try:
            # Luppet — собственный адрес
            self.luppet.send_message("/Luppet/BlendShape", [emotion, value])
            # VSeeFace — VMC Blend
            self.vseeface.send_message("/VMC/Ext/Blend/Apply", [emotion, value])
            # Unity — совместимо с VMC
            if self.enable_unity:
                self.unity.send_message("/VMC/Ext/Blend/Apply", [emotion, value])
            logger.debug(f"Emotion {emotion} → {value}")
        except Exception as e:
            logger.warning(f"send_emotion({emotion}) error: {e}")

    async def set_emotion(self, emotion_name: str, value: float = 1.0):
        """
        Асинхронная версия установки эмоции с маппингом названия → BlendShape.
        Пример: await avatar.set_emotion('happy', 1.0)
        """
        try:
            from .emotion import BLENDMAP
        except Exception:
            BLENDMAP = {}
        clip = BLENDMAP.get((emotion_name or "neutral").lower(), "Neutral")
        await asyncio.to_thread(self.send_emotion, clip, value)

    def pulse_emotion(self, emotion: str, intensity: float = 1.0, duration: float = 1.2):
        """Синхронная короткая эмоция (блокирует поток)"""
        try:
            self.send_emotion(emotion, intensity)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                loop.create_task(self.pulse_emotion_async(emotion, intensity, duration))
                return
            time.sleep(duration)
            self.send_emotion(emotion, 0.0)
        except Exception as e:
            logger.error(f"pulse_emotion error: {e}")

    async def pulse_emotion_async(self, emotion: str, intensity: float = 1.0, duration: float = 1.2):
        """Асинхронная короткая эмоция (не блокирует event loop)"""
        try:
            self.send_emotion(emotion, intensity)
            await asyncio.sleep(duration)
            self.send_emotion(emotion, 0.0)
        except Exception as e:
            logger.error(f"pulse_emotion_async error: {e}")

    # ============================== ПОЗЫ ==============================

    def send_pose(self, bone: str, position, rotation):
        """
        Отправляет позицию/поворот кости.
        - VSeeFace/Unity: /VMC/Ext/Bone/Pos
        - Luppet: MIDI (только тело, без головы), если мост активен
        """
        try:
            packet = [bone, *position, *rotation]
            self.vseeface.send_message("/VMC/Ext/Bone/Pos", packet)
            if self.enable_unity:
                self.unity.send_message("/VMC/Ext/Bone/Pos", packet)
        except Exception as e:
            logger.warning(f"send_pose VMC error: {e}")

        # Параллельно — MIDI в Luppet
        try:
            if self.luppet_bridge:
                self.luppet_bridge.send_pose(bone, position, rotation)
        except Exception:
            pass

    # ============================== LIP-SYNC ==============================

    def speak_signal(self, active: bool = True):
        """Передаёт сигнал 'говорит/молчит' для lipsync"""
        try:
            val = 1.0 if active else 0.0
            self.vseeface.send_message("/VMC/Ext/Talk", val)
            if self.enable_unity:
                self.unity.send_message("/Avatar/Talk", val)
            logger.debug(f"LipSync: {'ON' if active else 'OFF'}")
        except Exception as e:
            logger.error(f"speak_signal() error: {e}")

    # ============================== ЗАКРЫТИЕ ==============================

    def close(self):
        try:
            if self.luppet_bridge:
                self.luppet_bridge.close()
        except Exception:
            pass
                # Закрываем UDP-клиенты (SimpleUDPClient не имеет публичного close)
        for cli in (getattr(self, 'luppet', None), getattr(self, 'vseeface', None), getattr(self, 'unity', None)):
            if cli is not None:
                try:
                    sock = getattr(cli, '_sock', None)
                    if sock:
                        sock.close()
                except Exception:
                    pass
        logger.info("OSC/MIDI closed")


    async def shutdown(self):
        """Асинхронное закрытие для совместимости с системой."""
        try:
            await asyncio.to_thread(self.close)
        except Exception:
            pass
