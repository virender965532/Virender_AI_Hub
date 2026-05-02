"""Text-to-speech for interview questions via Hugging Face Inference (fal-ai)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 2500

MODEL_CHATTERBOX = "ResembleAI/chatterbox"
MODEL_KOKORO = "hexgrad/Kokoro-82M"

ALLOWED_MODELS = frozenset({MODEL_CHATTERBOX, MODEL_KOKORO})

CHATTERBOX_VOICES = frozenset(
    {
        "Aurora",
        "Blade",
        "Britney",
        "Carl",
        "Cliff",
        "Richard",
        "Rico",
        "Siobhan",
        "Vicky",
    }
)

KOKORO_VOICES = frozenset(
    {
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "bf_emma",
        "bf_isabella",
        "bm_george",
        "bm_lewis",
    }
)

_client = None


def hf_api_token() -> str:
    """HF Inference accepts the same token as the Hub CLI."""
    return (
        (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
        .strip()
    )


def _get_inference_client():
    global _client
    if _client is not None:
        return _client
    token = hf_api_token()
    if not token:
        return None
    from huggingface_hub import InferenceClient

    _client = InferenceClient(provider="fal-ai", api_key=token)
    return _client


def get_tts_options() -> Dict[str, Any]:
    """Metadata for the interview UI (models + voices)."""
    chatter_voices: List[Dict[str, str]] = [
        {"id": "", "label": "Random (surprise me)"},
    ]
    chatter_voices += [{"id": v, "label": v} for v in sorted(CHATTERBOX_VOICES)]

    kokoro_voices: List[Dict[str, str]] = [
        {"id": vid, "label": vid.replace("_", " · ", 1)} for vid in sorted(KOKORO_VOICES)
    ]

    return {
        "hf_token_configured": bool(hf_api_token()),
        "models": [
            {
                "id": MODEL_CHATTERBOX,
                "label": "Chatterbox (Resemble AI)",
                "hint": "Preset character voices; empty = random.",
                "voices": chatter_voices,
            },
            {
                "id": MODEL_KOKORO,
                "label": "Kokoro 82M",
                "hint": "Fast open-weight TTS; pick an English voice id.",
                "voices": kokoro_voices,
            },
        ],
        "defaultModel": MODEL_CHATTERBOX,
        "defaultVoiceByModel": {
            MODEL_CHATTERBOX: "",
            MODEL_KOKORO: "af_bella",
        },
    }


def _guess_audio_mime(audio: bytes) -> str:
    if len(audio) < 4:
        return "application/octet-stream"
    if audio[:4] == b"fLaC":
        return "audio/flac"
    if len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return "audio/wav"
    if audio[:4] == b"OggS":
        return "audio/ogg"
    if len(audio) >= 4 and audio[:4] == b"\x1a\x45\xdf\xa3":
        return "audio/webm"
    if len(audio) >= 12 and audio[4:8] == b"ftyp":
        return "audio/mp4"
    if audio[:3] == b"ID3":
        return "audio/mpeg"
    if audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    logger.warning(
        "TTS unknown audio signature (first 16 bytes hex): %s",
        audio[:16].hex(),
    )
    return "audio/wav"


def _extra_body_for_voice(model_id: str, voice: Optional[str]) -> Optional[dict]:
    v = (voice or "").strip()
    if model_id == MODEL_CHATTERBOX:
        if not v:
            return None
        if v not in CHATTERBOX_VOICES:
            logger.warning("Ignoring invalid Chatterbox voice %r", v)
            return None
        return {"voice": v}
    if model_id == MODEL_KOKORO:
        vid = v if v in KOKORO_VOICES else "af_bella"
        return {"voice": vid}
    return None


def synthesize_speech(
    text: str,
    *,
    model: Optional[str] = None,
    voice: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a dict: ok (bool), audio (bytes|None), mime (str|None), error (str|None).
    """
    out: Dict[str, Any] = {"ok": False, "audio": None, "mime": None, "error": None}
    text = (text or "").strip()
    if not text:
        out["error"] = "Empty text."
        return out

    token = hf_api_token()
    if not token:
        out["error"] = (
            "No Hugging Face token on the server. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your "
            ".env file (token needs Inference API / provider access), save the file, and fully restart "
            "the Flask process (not just refresh the browser)."
        )
        return out

    client = _get_inference_client()
    if not client:
        out["error"] = "Could not create InferenceClient."
        return out

    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]

    model_id = (model or MODEL_CHATTERBOX).strip()
    if model_id not in ALLOWED_MODELS:
        logger.warning("TTS model not allowed: %r — using Chatterbox", model_id)
        model_id = MODEL_CHATTERBOX

    extra = _extra_body_for_voice(model_id, voice)
    audio: Optional[bytes] = None
    last_err: Optional[BaseException] = None

    try:
        if extra:
            audio = client.text_to_speech(text, model=model_id, extra_body=extra)
        else:
            audio = client.text_to_speech(text, model=model_id)
    except Exception as e:
        last_err = e
        if model_id == MODEL_CHATTERBOX and extra:
            logger.warning(
                "Chatterbox TTS with preset voice failed (%s); retrying without voice parameter.",
                e,
            )
            try:
                audio = client.text_to_speech(text, model=model_id)
            except Exception as e2:
                last_err = e2
                audio = None
        else:
            audio = None

    if not audio:
        msg = str(last_err).strip() if last_err else "Unknown error"
        if not msg:
            msg = repr(last_err)
        logger.error("Interview TTS failed model=%s: %s", model_id, msg[:500])
        out["error"] = msg[:900]
        return out

    lead = audio.lstrip()[:1]
    if lead in (b"<", b"{", b"["):
        logger.error(
            "TTS returned non-audio payload (starts with %r): %r",
            lead,
            audio[:200],
        )
        out["error"] = "Provider returned JSON/HTML instead of audio. Check model billing and provider status."
        return out

    mime = _guess_audio_mime(audio)
    logger.info("TTS ok model=%s bytes=%s mime=%s", model_id, len(audio), mime)
    out["ok"] = True
    out["audio"] = audio
    out["mime"] = mime
    return out
