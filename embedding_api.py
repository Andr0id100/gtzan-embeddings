import torch

from models import MFCCModel
import tqdm

class EmbeddingAPI:
    def load_model(self, model_path):
        return torch.load(model_path)

    def get_timestamp_embeddings(self, audio, model):
        # Calculating timestamps separately by sliding window over the audio signal
        timestamps = []
        window_center = 0  # Staring from zero because padding allows window to go outside
        hop_length = model.mfcc.MelSpectrogram.hop_length

        audio_length = audio.shape[-1]
        while window_center <= audio_length:
            timestamps.append(window_center)
            window_center += hop_length

        timestamps = torch.tensor(timestamps)
        timestamps = (timestamps * 1000) / model.sample_rate
        timestamps = timestamps.repeat(audio.shape[0])

        embedding = model.get_embeddings(audio)
        if type(model) == MFCCModel:
            embedding = embedding.permute((0, 2, 1))

        return (embedding, timestamps)

    def get_scene_embeddings(self, audio, model, aggregation="mean"):
        assert aggregation in ("mean", "last")

        timestamp_embeddings, _ = self.get_timestamp_embeddings(audio, model)

        if aggregation == "mean":
            return timestamp_embeddings.mean(axis=1)
        elif aggregation == "last":
            return timestamp_embeddings[:, -1, :]
