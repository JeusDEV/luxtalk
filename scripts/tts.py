"""TTS module."""

import logging
import sys
import time

import soundfile as sf
import torch

from conf import logging_setup

LANGUAGE = "ru"
TTS_MODEL_ID = "v5_4_ru"
TTS_SPEAKERS = ["aidar", "baya", "kseniya", "xenia"]

tts_speaker: str = TTS_SPEAKERS[0]

logging_setup()
logger = logging.getLogger(__name__)

device = torch.device("cuda")

model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language=LANGUAGE,
    speaker=TTS_MODEL_ID,
    device=device,
)
logger.info("(i) TTS model loaded")

def tts_inference(
    output_path: str,
    speaker: str,
    text: str = "В недрах тундры выдры в гетрах тырят в вёдра ядра кедров.",
) -> None:
    """Synthesizes speech from text with SileroTTS model and saves as an audio file."""
    audio = model.apply_tts(
        text=text,
        speaker=speaker,
        sample_rate=48000,
    ).numpy().squeeze()
    logger.info("(i) TTS completed")

    sf.write(output_path, audio, 48000)
    logger.info(f"(V) TTS saved as '{output_path}'")


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        index = str(args[1])
        if index in TTS_SPEAKERS:
            tts_speaker = index
        else:
            logger.warning(f"(!) Speaker '{index}' is unavailable (-> default)")
            logger.info(f"(i) Available speakers: {TTS_SPEAKERS}")
    logger.info(f"(i) Speaker '{tts_speaker}' was selected for TTS")

    output_path = f"./media/output/{str(int(time.time()))[-7:]}_{tts_speaker}.mp3"

    logger.info("(i) Processing TTS")
    tts_inference(output_path=output_path, speaker=tts_speaker)
    logger.info("(V) TTS completed & saved")
