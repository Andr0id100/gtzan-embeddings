from torch import nn
import torchaudio


class MFCCModel(nn.Module):
    def __init__(self, sample_rate=22050, embedding_size=64):
        super(MFCCModel, self).__init__()
        # Simply model that provides MFCC features, no learnable parameters
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=embedding_size)

        self.sample_rate = sample_rate
        self.scene_embedding_size = embedding_size
        self.timestamp_embedding_size = embedding_size

    def forward(self, audios):
        return self.mfcc(audios)

    def get_embeddings(self, audio):
        return self.mfcc(audio).detach()

class LSTMModel(nn.Module):
    def __init__(self, sample_rate=22050, embeddings_size=64, num_classes=10, last=True):
        super(LSTMModel, self).__init__()
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)
        self.lstm = nn.LSTM(input_size=self.mfcc.n_mfcc,
                            hidden_size=embeddings_size)
        self.linear = nn.Linear(
            in_features=embeddings_size, out_features=num_classes)

        self.last = last

        self.sample_rate = sample_rate
        self.timestamp_embedding_size = embeddings_size
        self.scene_embedding_size = embeddings_size


    def forward(self, audios):
        batch_size = audios.shape[0]

        mfcc_features = self.mfcc(audios).squeeze(
            1).reshape(batch_size, -1, self.mfcc.n_mfcc)
        (hidden_states, _) = self.lstm(mfcc_features)

        if self.last:
            classified = self.linear(hidden_states[:, -1, :])
        else:
            classified = self.linear(hidden_states.mean(axis=1))

        return classified

    def get_embeddings(self, audios):
        batch_size = audios.shape[0]

        mfcc_features = self.mfcc(audios).squeeze(
            1).reshape(batch_size, -1, self.mfcc.n_mfcc)
        (hidden_states, _) = self.lstm(mfcc_features)
        return hidden_states.detach()