
from __future__ import annotations
import asyncio
import contextlib
import math
from typing import Optional, Callable, Deque, Tuple
from collections import deque

import numpy as np

# --- Optional deps ---
try:
    import onnxruntime as ort  # for Silero VAD (ONNX)
except Exception:
    ort = None

try:
    import webrtcvad  # fallback VAD
except Exception:
    webrtcvad = None

try:
    from faster_whisper import WhisperModel  # ASR backend
except Exception:
    WhisperModel = None

try:
    import sounddevice as sd  # microphone capture
except Exception:
    sd = None


def _dbfs(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    if x.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(np.square(x / 32768.0)))
    if rms <= 1e-12:
        return -120.0
    return 20.0 * math.log10(rms + 1e-12)


class _SileroONNXVAD:
    """Tiny Silero VAD (ONNX) wrapper. Returns probability of speech for a window."""
    def __init__(self, model_path: str, sample_rate: int = 16000):
        self.sr = sample_rate
        self.sess = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.inp = self.sess.get_inputs()[0].name

    def prob(self, mono_pcm16: np.ndarray) -> float:
        if mono_pcm16.dtype == np.int16:
            x = mono_pcm16.astype(np.float32) / 32768.0
        else:
            x = mono_pcm16.astype(np.float32)
        x = x[None, :]  # [1, T]
        p = float(self.sess.run(None, {self.inp: x})[0].squeeze())
        return float(np.clip(p, 0.0, 1.0))


class _WebRTCVAD:
    """Simple WebRTC VAD wrapper. Returns ratio of voiced frames in a window."""
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2):
        self.sr = sample_rate
        self.v = webrtcvad.Vad(int(np.clip(aggressiveness, 0, 3)))
        self.frame_ms = 30
        self.frame_len = int(self.sr * self.frame_ms / 1000)

    def prob(self, mono_pcm16: np.ndarray) -> float:
        if mono_pcm16.dtype != np.int16:
            arr = (mono_pcm16.astype(np.float32) * 32768.0).astype(np.int16)
        else:
            arr = mono_pcm16
        total, voiced = 0, 0
        step = self.frame_len
        for i in range(0, max(0, len(arr) - self.frame_len + 1), step):
            frame = arr[i:i+self.frame_len].tobytes()
            try:
                if self.v.is_speech(frame, self.sr):
                    voiced += 1
            except Exception:
                pass
            total += 1
        return (voiced / total) if total else 0.0


class WhisperSTT:
    """Asynchronous STT with VAD → faster-whisper.
    - Low VRAM defaults (model_size='small', compute_type='float16' on GPU, 'int8_float16' on CPU).
    - Returns final text via await listen(); optional callbacks for partials / metrics.
    """
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "float16",
        beam_size: int = 1,
        sample_rate: int = 16000,
        chunk_sec: float = 0.5,
        # VAD
        vad_backend: str = "silero_onnx",
        silero_model_path: Optional[str] = None,
        speech_threshold: float = 0.55,
        silence_threshold: float = 0.45,
        min_speech_ms: int = 200,
        min_silence_ms: int = 500,
        # callbacks
        on_partial: Optional[Callable[[str], None]] = None,
        on_metrics: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.sr = sample_rate
        self.chunk_len = int(chunk_sec * sample_rate)
        self.on_partial = on_partial
        self.on_metrics = on_metrics


        # VAD thresholds and hysteresis windows
        self._thr_speech = float(speech_threshold)
        self._thr_sil = float(silence_threshold)
        # convert ms to sample counts for duration gating
        self._min_speech = int(max(0, (min_speech_ms / 1000.0) * self.sr))
        self._min_sil = int(max(0, (min_silence_ms / 1000.0) * self.sr))
        # running counters (in samples)
        self._speech_dur = 0
        self._sil_dur = 0

        # VAD init
        self._vad = None
        if vad_backend == "silero_onnx" and silero_model_path and ort is not None:
            try:
                self._vad = _SileroONNXVAD(silero_model_path, sample_rate)
            except Exception:
                self._vad = None
        if self._vad is None and webrtcvad is not None:
            self._vad = _WebRTCVAD(sample_rate, aggressiveness=2)
        if self._vad is None:
            raise RuntimeError("No VAD backend available (Silero ONNX or WebRTC).")

        # ASR init
        dev = "cpu"
        if device in ("cuda", "auto"):
            try:
                import torch
                if torch.cuda.is_available():
                    dev = "cuda"
                else:
                    dev = "cpu"
            except Exception:
                dev = "cpu"

        ctype = compute_type
        if dev == "cpu" and compute_type == "float16":
            ctype = "int8_float16"

        if WhisperModel is None:
            raise RuntimeError("faster-whisper not installed")
        self._asr = WhisperModel(model_size, device=dev, compute_type=ctype)
        self._beam = beam_size

        # audio capture state
        self._speaking = False
        self._speech_buf: Deque[np.ndarray] = deque()
        self._queue: Deque[np.ndarray] = deque()
        self._closed = False

        # concurrency
        self._lock = asyncio.Lock()

    # -------------------- public API --------------------
    async def listen(self) -> str:
        """Capture mic until end-of-speech and return final recognized text."""
        if sd is None:
            raise RuntimeError("sounddevice is not available for microphone capture")

        # reset state
        self._speaking = False
        self._speech_buf.clear()
        self._queue.clear()

        stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="int16", blocksize=self.chunk_len)
        with stream:
            final_text = ""
            # background ASR pump
            asr_task = asyncio.create_task(self._asr_loop())

            try:
                while True:
                    await asyncio.sleep(0)  # yield
                    frames, overflowed = stream.read(self.chunk_len)  # numpy int16
                    if overflowed and self.on_metrics:
                        try:
                            self.on_metrics(float('nan'))
                        except Exception:
                            pass
                    chunk = frames.reshape(-1).copy()

                    # metrics (optional)
                    if self.on_metrics:
                        self.on_metrics(_dbfs(chunk))

                    is_speech, ev = self._vad_step(chunk)

                    if is_speech:
                        # send chunk(s) to ASR queue
                        self._queue.append(chunk)
                    else:
                        if ev == "speech_end":
                            # Wait a short tail to ensure ASR flushed
                            await asyncio.sleep(0.01)
                            final_text = await self._flush_and_get_final()
                            break
            finally:
                self._closed = True
                with contextlib.suppress(Exception):
                    asr_task.cancel()

        return final_text.strip()

    # -------------------- internals --------------------
    def _vad_step(self, chunk: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Process chunk through VAD state machine with hysteresis.
        Returns (is_speech_now, event) where event is 'speech_end' when switching to silence.
        """
        p = self._vad.prob(chunk)
        event = None
        self._speech_buf.append(chunk)

        # speaking state -> wait for sustained silence
        if self._speaking:
            if p < self._thr_sil:
                self._sil_dur += len(chunk)
                if self._sil_dur >= self._min_sil:
                    self._speaking = False
                    self._sil_dur = 0
                    self._speech_dur = 0
                    event = "speech_end"
            else:
                self._sil_dur = 0
        else:
            # silence state -> wait for sustained speech
            if p >= self._thr_speech:
                self._speech_dur += len(chunk)
                if self._speech_dur >= self._min_speech:
                    self._speaking = True
                    self._speech_dur = 0
            else:
                self._speech_dur = 0

        return self._speaking, event

    async def _asr_loop(self) -> None:
        """Consume queue and push partial hypotheses via callback."""
        # We aggregate a few chunks for stability
        agg: Deque[np.ndarray] = deque()
        sec_acc = 0.0
        while not self._closed:
            if not self._queue:
                await asyncio.sleep(0.01)
                continue
            chunk = self._queue.popleft()
            agg.append(chunk)
            sec_acc += len(chunk) / float(self.sr)

            if sec_acc >= max(0.5, self.chunk_len / self.sr):
                audio = np.concatenate(list(agg))
                agg.clear()
                sec_acc = 0.0

                # Run ASR
                segments, _ = self._asr.transcribe(audio, beam_size=self._beam, vad_filter=None)
                text = " ".join(s.text for s in segments).strip()
                if text and self.on_partial:
                    # fire-and-forget (no await bottleneck)
                    try:
                        self.on_partial(text)
                    except Exception:
                        pass

    async def _flush_and_get_final(self) -> str:
        """Flush remaining audio and do a final pass for better quality."""
        # Mix any remaining queued + speech_buf
        pending = list(self._queue) + list(self._speech_buf)
        self._queue.clear()
        self._speech_buf.clear()
        if not pending:
            return ""
        audio = np.concatenate(pending)
        segments, _ = self._asr.transcribe(audio, beam_size=self._beam, vad_filter=None)
        return " ".join(s.text for s in segments).strip()
