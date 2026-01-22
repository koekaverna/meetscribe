"""Voice Activity Detection using SpeechBrain."""

from pathlib import Path
from dataclasses import dataclass

import torch
import torchaudio
from speechbrain.inference.VAD import VAD
from speechbrain.utils.fetching import LocalStrategy


@dataclass
class SpeechSegment:
    """A segment of detected speech."""

    start_ms: int
    end_ms: int
    cluster_id: int | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class VADProcessor:
    """Voice Activity Detection using SpeechBrain.

    Uses CRDNN model with energy-based double-checking to reduce hallucinations.
    Parameters follow WhisperX-style naming conventions.
    """

    MODEL_SOURCE = "speechbrain/vad-crdnn-libriparty"
    SAMPLE_RATE = 16000

    # VAD parameters (WhisperX-style naming)
    VAD_ONSET = 0.5  # Speech start threshold (0-1), higher = stricter
    VAD_OFFSET = 0.25  # Speech end threshold, lower = less likely to cut off speech
    MIN_SPEECH_MS = 250  # Minimum speech segment duration (ms)
    MIN_SILENCE_MS = 250  # Minimum silence to split segments (ms)

    # SpeechBrain-specific
    LARGE_CHUNK_SIZE = 30  # Seconds, for initial pass
    SMALL_CHUNK_SIZE = 10  # Seconds, for refinement
    APPLY_ENERGY_VAD = True  # Energy-based double-check (reduces hallucinations)

    def __init__(self, device: str = "cuda", cache_dir: Path | None = None):
        self.device = device
        savedir = str(cache_dir / "vad") if cache_dir else "pretrained_models/vad"

        self.model = VAD.from_hparams(
            source=self.MODEL_SOURCE,
            savedir=savedir,
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY,
        )

    def process(self, audio_path: Path, min_duration_ms: int | None = None) -> list[SpeechSegment]:
        """Detect speech segments in audio file.

        Args:
            audio_path: Path to audio file
            min_duration_ms: Minimum segment duration (default: MIN_SPEECH_MS)

        Returns:
            List of detected speech segments
        """
        if min_duration_ms is None:
            min_duration_ms = self.MIN_SPEECH_MS

        boundaries = self.model.get_speech_segments(
            audio_path.as_posix(),
            large_chunk_size=self.LARGE_CHUNK_SIZE,
            small_chunk_size=self.SMALL_CHUNK_SIZE,
            overlap_small_chunk=True,
            apply_energy_VAD=self.APPLY_ENERGY_VAD,
            double_check=True,
            close_th=self.MIN_SILENCE_MS / 1000,  # Convert to seconds
            len_th=self.MIN_SPEECH_MS / 1000,  # Convert to seconds
        )

        segments = []
        for boundary in boundaries:
            start_sec, end_sec = boundary[0].item(), boundary[1].item()
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)

            if (end_ms - start_ms) >= min_duration_ms:
                segments.append(SpeechSegment(start_ms=start_ms, end_ms=end_ms))

        return segments

    def extract_segment_audio(self, audio_path: Path, segment: SpeechSegment) -> torch.Tensor:
        """Extract audio tensor for a specific segment."""
        waveform, sr = torchaudio.load(str(audio_path))

        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        start_sample = int(segment.start_ms * self.SAMPLE_RATE / 1000)
        end_sample = int(segment.end_ms * self.SAMPLE_RATE / 1000)

        return waveform[:, start_sample:end_sample]
