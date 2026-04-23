"""STT module."""

import logging

import librosa
import torch
from torch import Tensor

from conf import logging_setup

FLOAT_TOLERANCE = 1e-6
STT_MODEL_ID = "ru"  # "en"
TARGET_PATH = "./media/input/someonecat.mp3"

logging_setup()
logger = logging.getLogger(__name__)

device = torch.device("cuda")

(model,
    decoder,
    (_, split_into_batches, _, prepare_model_input),
) = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_stt",
    language=STT_MODEL_ID,
    device=device,
)
logger.info("(i) STT model loaded")


def read_audio_soundfile(path: str, target_sr: int = 16000) -> Tensor:
    """Convert stereo to mono and resample to 16kHz."""
    wav, _ = librosa.load(
        path,
        sr=target_sr,
        mono=True,
    )
    return torch.from_numpy(wav).float()


def read_batch(paths: list) -> list:
    return [read_audio_soundfile(p) for p in paths]


def stt_inference(target_path: str) -> str:
    """Transcribes audio files into text using the Silero STT model."""

    batches = split_into_batches([target_path], batch_size=10)

    wav_data = read_batch(batches[0])[0]

    logger.info("(d) About media\n" +
        (f"Audio shape: {wav_data.shape}, dtype: {wav_data.dtype}\n"
        f"Min: {wav_data.min():.4f}, Max: {wav_data.max():.4f}, Mean: {wav_data.mean():.4f}\n"
        f"Non-zero samples: {(wav_data.abs() > FLOAT_TOLERANCE).sum()}\n"),
    )

    inp = prepare_model_input(read_batch(batches[0]), device=device)

    output = model(inp)
    logger.info(f"(d) Output tensor:\n{output}")
    logger.info("(i) STT completed")

    return "".join(decoder(example.cpu()) for example in output)


if __name__ == "__main__":
    logger.info("(i) Processing STT")
    result = stt_inference(target_path=TARGET_PATH)
    logger.info(f"(V) {result}")
