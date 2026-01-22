"""Speaker embedding extraction using SpeechBrain ECAPA-TDNN."""

from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


class EmbeddingExtractor:
    """Extract speaker embeddings using ECAPA-TDNN."""

    MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
    SAMPLE_RATE = 16000

    def __init__(self, device: str = "cuda", cache_dir: Path | None = None):
        self.device = device
        savedir = str(cache_dir / "embeddings") if cache_dir else "pretrained_models/embeddings"

        self.model = EncoderClassifier.from_hparams(
            source=self.MODEL_SOURCE,
            savedir=savedir,
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY
        )

    def extract_from_file(self, audio_path: Path) -> np.ndarray:
        """Extract embedding from audio file."""
        waveform, sr = torchaudio.load(str(audio_path))

        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return self._extract(waveform)

    def extract_from_tensor(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract embedding from audio tensor."""
        return self._extract(waveform)

    def _extract(self, waveform: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            embedding = self.model.encode_batch(waveform.to(self.device))
            return embedding.squeeze().cpu().numpy()

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
