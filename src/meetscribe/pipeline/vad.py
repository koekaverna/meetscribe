"""Voice Activity Detection using SpeechBrain."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from speechbrain.inference.VAD import VAD
from speechbrain.utils.fetching import LocalStrategy
from tqdm import tqdm


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

    def get_audio_duration_sec(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        info = torchaudio.info(str(audio_path))
        return info.num_frames / info.sample_rate

    def process(
        self,
        audio_path: Path,
        min_duration_ms: int | None = None,
        show_progress: bool = False,
    ) -> list[SpeechSegment]:
        """Detect speech segments in audio file.

        Args:
            audio_path: Path to audio file
            min_duration_ms: Minimum segment duration (default: MIN_SPEECH_MS)
            show_progress: Show progress bar

        Returns:
            List of detected speech segments
        """
        if min_duration_ms is None:
            min_duration_ms = self.MIN_SPEECH_MS

        duration_sec = self.get_audio_duration_sec(audio_path)
        num_chunks = max(1, int(duration_sec / self.LARGE_CHUNK_SIZE))

        if show_progress and num_chunks > 1:
            # Process in chunks with progress bar
            chunk_size_sec = self.LARGE_CHUNK_SIZE
            waveform, sr = torchaudio.load(str(audio_path))

            if sr != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
                waveform = resampler(waveform)
                sr = self.SAMPLE_RATE

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            total_samples = waveform.shape[1]
            chunk_samples = int(chunk_size_sec * sr)
            all_segments = []

            pbar = tqdm(
                range(0, total_samples, chunk_samples),
                desc="    VAD",
                leave=False,
                unit="chunk",
            )
            for start_sample in pbar:
                end_sample = min(start_sample + chunk_samples, total_samples)
                chunk = waveform[:, start_sample:end_sample]
                offset_ms = int(start_sample * 1000 / sr)

                # Save chunk to temp file for processing
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                torchaudio.save(str(tmp_path), chunk, sr)

                try:
                    boundaries = self.model.get_speech_segments(
                        tmp_path.as_posix(),
                        large_chunk_size=self.LARGE_CHUNK_SIZE,
                        small_chunk_size=self.SMALL_CHUNK_SIZE,
                        overlap_small_chunk=True,
                        apply_energy_VAD=self.APPLY_ENERGY_VAD,
                        double_check=True,
                        close_th=self.MIN_SILENCE_MS / 1000,
                        len_th=self.MIN_SPEECH_MS / 1000,
                    )

                    for boundary in boundaries:
                        start_sec, end_sec = boundary[0].item(), boundary[1].item()
                        start_ms = int(start_sec * 1000) + offset_ms
                        end_ms = int(end_sec * 1000) + offset_ms

                        if (end_ms - start_ms) >= min_duration_ms:
                            all_segments.append(SpeechSegment(start_ms=start_ms, end_ms=end_ms))
                finally:
                    tmp_path.unlink(missing_ok=True)

            return self._merge_adjacent_segments(all_segments, min_duration_ms)

        # Single pass for short audio or when progress not needed
        boundaries = self.model.get_speech_segments(
            audio_path.as_posix(),
            large_chunk_size=self.LARGE_CHUNK_SIZE,
            small_chunk_size=self.SMALL_CHUNK_SIZE,
            overlap_small_chunk=True,
            apply_energy_VAD=self.APPLY_ENERGY_VAD,
            double_check=True,
            close_th=self.MIN_SILENCE_MS / 1000,
            len_th=self.MIN_SPEECH_MS / 1000,
        )

        segments = []
        for boundary in boundaries:
            start_sec, end_sec = boundary[0].item(), boundary[1].item()
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)

            if (end_ms - start_ms) >= min_duration_ms:
                segments.append(SpeechSegment(start_ms=start_ms, end_ms=end_ms))

        return segments

    def _merge_adjacent_segments(
        self, segments: list[SpeechSegment], min_duration_ms: int
    ) -> list[SpeechSegment]:
        """Merge adjacent segments that may have been split at chunk boundaries."""
        if not segments:
            return segments

        segments.sort(key=lambda s: s.start_ms)
        merged = [segments[0]]

        for seg in segments[1:]:
            last = merged[-1]
            # Merge if gap is less than MIN_SILENCE_MS
            if seg.start_ms - last.end_ms < self.MIN_SILENCE_MS:
                merged[-1] = SpeechSegment(start_ms=last.start_ms, end_ms=seg.end_ms)
            else:
                merged.append(seg)

        return [s for s in merged if s.duration_ms >= min_duration_ms]

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
