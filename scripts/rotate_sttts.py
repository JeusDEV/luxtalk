"""STT -> TTS rotation test module."""

import logging
import time

import stt
import tts
from conf import logging_setup

logging_setup()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("(i) Processing STT")
    result = stt.stt_inference(target_path=stt.TARGET_PATH)
    logger.info(f"(V) {result}")

    tts_speaker = tts.TTS_SPEAKERS[0]
    output_path = f"./media/output/{str(int(time.time()))[-7:]}_{tts_speaker}.mp3"

    logger.info("(i) Processing TTS")
    tts.tts_inference(output_path=output_path, speaker=tts_speaker, text=result)
    logger.info("(V) TTS completed & saved")
