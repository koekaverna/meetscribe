"""Audio transcription using OpenAI Whisper."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


def merge_close_segments(
    vad_segments: list[tuple[int, int]], max_gap_ms: int = 500, max_chunk_ms: int = 30000
) -> list[tuple[int, int]]:
    """Merge VAD segments with small gaps into larger chunks."""
    if not vad_segments:
        return []

    merged = []
    current_start, current_end = vad_segments[0]

    for start, end in vad_segments[1:]:
        gap = start - current_end
        duration = end - current_start

        if gap <= max_gap_ms and duration <= max_chunk_ms:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def is_silent(waveform: torch.Tensor, threshold_db: float = -40.0) -> bool:
    """Check if audio segment is silent based on RMS energy."""
    rms = torch.sqrt(torch.mean(waveform**2))

    if rms < 1e-10:
        return True

    db = 20 * torch.log10(rms)
    return db.item() < threshold_db


@dataclass
class TranscriptSegment:
    """Transcribed segment with timestamp."""

    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None


class Transcriber:
    """Transcription using openai-whisper (PyTorch)."""

    DEFAULT_MODEL = "medium"
    DEFAULT_LANGUAGE = "ru"

    BEAM_SIZE = 5
    CONDITION_ON_PREVIOUS_TEXT = False
    TEMPERATURE = 0.0
    NO_SPEECH_THRESHOLD = 0.5
    COMPRESSION_RATIO_THRESHOLD = 2.4
    HALLUCINATION_SILENCE_THRESHOLD = 2.0
    SILENCE_THRESHOLD_DB = -40.0

    def __init__(self, model_size: str = DEFAULT_MODEL, device: str = "cuda"):
        self.model_size = model_size
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self):
        import whisper

        self._model = whisper.load_model(self.model_size, device=self.device)

    def transcribe(
        self,
        audio_path: Path,
        language: str = DEFAULT_LANGUAGE,
        speaker: str | None = None,
    ) -> list[TranscriptSegment]:
        """Transcribe audio file."""
        result = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self.BEAM_SIZE,
            condition_on_previous_text=self.CONDITION_ON_PREVIOUS_TEXT,
            temperature=self.TEMPERATURE,
            no_speech_threshold=self.NO_SPEECH_THRESHOLD,
            compression_ratio_threshold=self.COMPRESSION_RATIO_THRESHOLD,
            verbose=None,
            word_timestamps=True,
            hallucination_silence_threshold=self.HALLUCINATION_SILENCE_THRESHOLD,
        )

        segments = []
        for seg in result["segments"]:
            text = seg["text"].strip()
            if text:
                segments.append(
                    TranscriptSegment(
                        start_ms=int(seg["start"] * 1000),
                        end_ms=int(seg["end"] * 1000),
                        text=text,
                        speaker=speaker,
                    )
                )

        return segments

    def transcribe_vad_segments(
        self,
        audio_path: Path,
        vad_segments: list[tuple[int, int]],
        speaker_segments: list[tuple[int, int, str]],
        language: str = DEFAULT_LANGUAGE,
    ) -> list[TranscriptSegment]:
        """Transcribe only speech segments detected by VAD."""
        merged_segments = merge_close_segments(vad_segments)

        if not merged_segments:
            return []

        waveform, sr = torchaudio.load(str(audio_path))
        results = []

        # Calculate total duration for progress bar
        total_ms = sum(end - start for start, end in merged_segments)
        pbar = tqdm(total=total_ms, unit="ms", unit_scale=True, desc="  Transcribing", leave=False)

        for vad_start, vad_end in merged_segments:
            start_sample = int(vad_start * sr / 1000)
            end_sample = int(vad_end * sr / 1000)
            chunk = waveform[:, start_sample:end_sample]
            chunk_duration = vad_end - vad_start

            if chunk.shape[1] < sr * 0.1:
                pbar.update(chunk_duration)
                continue

            # Skip silent chunks (prevents hallucinations)
            if is_silent(chunk, threshold_db=self.SILENCE_THRESHOLD_DB):
                pbar.update(chunk_duration)
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                torchaudio.save(f.name, chunk, sr)
                chunk_path = Path(f.name)

            try:
                segments = self.transcribe(chunk_path, language)

                for seg in segments:
                    seg.start_ms += vad_start
                    seg.end_ms += vad_start
                    seg.speaker = self._find_speaker(seg.start_ms, seg.end_ms, speaker_segments)

                results.extend(segments)
            finally:
                chunk_path.unlink(missing_ok=True)

            pbar.update(chunk_duration)

        pbar.close()
        return results

    def _find_speaker(
        self, start_ms: int, end_ms: int, time_segments: list[tuple[int, int, str]]
    ) -> str:
        """Find speaker with maximum time overlap."""
        best_speaker = "Unknown"
        best_overlap = 0

        for seg_start, seg_end, speaker in time_segments:
            overlap = max(0, min(end_ms, seg_end) - max(start_ms, seg_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        return best_speaker
