"""Minimal audio synthesis: render a :class:`Riff` to a WAV byte string.

No external audio libraries, just numpy.  The sound is a plucked-string-ish
additive tone with exponential decay, which is enough to hear whether a
reconstruction matches the riff in the user's head.  It is deliberately not a
realistic guitar: the plan says not to spend time there yet.
"""

from __future__ import annotations

import io
import math
import wave

import numpy as np

from .representation import Riff

SR = 22050


def midi_to_hz(p: float) -> float:
    return 440.0 * 2.0 ** ((p - 69.0) / 12.0)


def _pluck(freq_curve: np.ndarray, sr: int, decay: float, brightness: float,
           n_harm: int = 8) -> np.ndarray:
    """Additive pluck. ``freq_curve`` is per-sample fundamental in Hz."""
    n = len(freq_curve)
    t = np.arange(n) / sr
    phase = 2 * np.pi * np.cumsum(freq_curve) / sr
    env = np.exp(-t / decay)
    out = np.zeros(n)
    for k in range(1, n_harm + 1):
        amp = brightness ** (k - 1) / k
        # higher harmonics decay faster, as on a real string
        out += amp * np.sin(k * phase) * np.exp(-t * (k - 1) * 2.0 / decay)
    out *= env
    # short attack to avoid clicks
    atk = min(n, int(0.004 * sr))
    if atk > 0:
        out[:atk] *= np.linspace(0, 1, atk)
    return out


def render(riff: Riff, sr: int = SR, tail_s: float = 0.6, gain: float = 0.5) -> np.ndarray:
    """Render the riff to a float32 mono signal in [-1, 1]."""
    if not riff.notes:
        return np.zeros(int(sr * 0.5), dtype=np.float32)
    spb = 60.0 / riff.tempo  # seconds per beat
    end_s = max(n.onset + n.duration for n in riff.notes) * spb + tail_s
    out = np.zeros(int(end_s * sr) + 1)
    for idx, note in enumerate(riff.notes):
        art = note.articulation
        start = int(note.onset * spb * sr)
        dur_s = note.duration * spb
        if art.get("staccato"):
            dur_s *= 0.5
        if art.get("palm_mute"):
            dur_s = min(dur_s, 0.18)
        # let the note ring a bit past its nominal end
        length = int((dur_s + 0.35) * sr)
        length = min(length, len(out) - start)
        if length <= 0:
            continue
        f0 = midi_to_hz(note.pitch)
        freq = np.full(length, f0)
        tt = np.arange(length) / sr
        if art.get("bend"):
            semis = float(art["bend"])
            bend_time = min(dur_s * 0.6, 0.25)
            ramp = np.clip(tt / max(bend_time, 1e-3), 0, 1)
            freq = f0 * 2.0 ** (semis * ramp / 12.0)
        elif art.get("slide") and idx + 1 < len(riff.notes):
            target = midi_to_hz(riff.notes[idx + 1].pitch)
            ramp = np.clip((tt - dur_s * 0.5) / max(dur_s * 0.5, 1e-3), 0, 1)
            freq = f0 * (target / f0) ** ramp
        if art.get("vibrato"):
            freq = freq * 2.0 ** (0.3 * np.sin(2 * np.pi * 5.5 * tt) / 12.0)
        decay = 0.12 if art.get("palm_mute") else 0.9
        brightness = 0.45 if art.get("palm_mute") else 0.7
        sig = _pluck(freq, sr, decay, brightness)
        vel = note.velocity / 127.0
        out[start:start + length] += sig * vel
    out = np.tanh(out * gain * 1.5)  # soft clip
    return out.astype(np.float32)


def to_wav_bytes(samples: np.ndarray, sr: int = SR) -> bytes:
    pcm = np.clip(samples, -1, 1)
    pcm16 = (pcm * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def riff_to_wav(riff: Riff, sr: int = SR) -> bytes:
    return to_wav_bytes(render(riff, sr), sr)


def write_wav(path: str, riff: Riff, sr: int = SR) -> None:
    with open(path, "wb") as f:
        f.write(riff_to_wav(riff, sr))
