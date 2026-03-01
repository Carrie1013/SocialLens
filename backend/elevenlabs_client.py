"""
ElevenLabs TTS client: generates audio for person voice profiles.
"""

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice mapping: (gender_presentation, personality_impression) → ElevenLabs voice ID
# Populate with your real ElevenLabs voice IDs after reviewing the library.
# These are placeholder IDs from the ElevenLabs premade voice library.
# ---------------------------------------------------------------------------

VOICE_MAP: dict[str, str] = {
    # Masculine voices
    "masculine_confident":   "pNInz6obpgDQGcFmaJgB",   # Adam
    "masculine_calm":        "VR6AewLTigWG4xSOukaG",   # Arnold
    "masculine_energetic":   "yoZ06aMxZJJ28mfd3POQ",   # Sam
    "masculine_authoritative":"nPczCjzI2devNBz1zQrb",  # Brian
    "masculine_playful":     "SOYHLrjzK2X1ezoPC6cr",   # Harry
    "masculine_shy":         "flq6f7yk4E4fJM5XTYuZ",   # Michael
    # Feminine voices
    "feminine_confident":    "EXAVITQu4vr4xnSDxMaL",   # Bella
    "feminine_calm":         "ThT5KcBeYPX3keUQqHPh",   # Dorothy
    "feminine_energetic":    "jsCqWAovK2LkecY7zXl4",   # Freya
    "feminine_authoritative":"XB0fDUnXU5powFXDhCwa",   # Charlotte
    "feminine_playful":      "oWAxZDx7w5VEj9dCyTzz",   # Grace
    "feminine_shy":          "jBpfuIE2acCO8z3wKNLl",   # Gigi
    # Androgynous / fallback
    "androgynous_calm":      "21m00Tcm4TlvDq8ikWAM",   # Rachel
    "androgynous_energetic": "AZnzlk1XvdvUeBnXmlld",   # Domi
    "androgynous_confident": "MF3mGyEYCl7XYWbV9V6O",   # Elli
    "androgynous_playful":   "IKne3meq5aSn9XLyUdCD",   # Charlie
    "androgynous_measured":  "21m00Tcm4TlvDq8ikWAM",   # Rachel (default)
    "androgynous_monotone":  "21m00Tcm4TlvDq8ikWAM",   # Rachel (default)
}

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"   # Rachel — safe fallback


def select_voice_id(gender_presentation: str, personality_impression: str) -> str:
    """Pick the best matching ElevenLabs voice ID."""
    key = f"{gender_presentation}_{personality_impression}"
    return VOICE_MAP.get(key, DEFAULT_VOICE_ID)


class ElevenLabsClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None
        if api_key:
            try:
                import elevenlabs
                self._client = elevenlabs.ElevenLabs(api_key=api_key)
                logger.info("ElevenLabs client initialized.")
            except ImportError:
                logger.warning("elevenlabs package not installed — TTS disabled.")
            except Exception as e:
                logger.error("ElevenLabs init error: %s", e)

    def is_available(self) -> bool:
        return self._client is not None

    def generate_audio(
        self,
        text: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
    ) -> Optional[bytes]:
        """
        Generate TTS audio and return raw bytes (MP3).
        Returns None if unavailable or on error.
        """
        if not self._client:
            logger.warning("ElevenLabs client not available.")
            return None

        try:
            audio_generator = self._client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                voice_settings={
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
            )
            # Convert generator/bytes to bytes
            if hasattr(audio_generator, "__iter__") and not isinstance(audio_generator, bytes):
                buf = io.BytesIO()
                for chunk in audio_generator:
                    buf.write(chunk)
                return buf.getvalue()
            return audio_generator
        except Exception as e:
            logger.error("ElevenLabs TTS error: %s", e)
            return None

    async def generate_audio_async(
        self,
        text: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
    ) -> Optional[bytes]:
        """Async wrapper — runs synchronous ElevenLabs call in thread pool."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate_audio, text, voice_id, model_id
        )
