from webbrowser import get
import torchaudio
from tqdm import tqdm
from fastcore.xtras import Path
from torch.utils.data import DataLoader

# Defined here to simplify testing
labels = ['blues', 'classical', 'country', 'disco',
          'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
int2label = labels[:]
label2int = {x: i for (i, x) in enumerate(labels)}


def get_file_names(data_root):
    genre_directories = data_root.ls()
    file_paths = []
    for genre in genre_directories:
        file_paths.extend(genre.ls())

    return file_paths


def read_audio_and_genre(file_path):
    # Assuming a fixed sampling rate of 22050
    audio, _ = torchaudio.load(file_path)
    genre = file_path.parts[-2]
    return audio, genre


def load_data(file_paths):
    audios, genres = [], []

    for file_path in tqdm(file_paths):
        audio, genre = read_audio_and_genre(file_path)
        # All audio clips trimmed to the shortest length to simplify processing
        # This amounts to <1 sec of maximum loss of information
        audios.append(audio[:, :660000])
        genres.append(genre)

    return (audios, genres)


def load_split(split_path):
    with open(split_path) as f:
        file_paths = f.read().split("\n")

    # Remove empty line at end
    file_paths.pop(-1)

    file_paths = [Path(x) for x in file_paths]

    return load_data(file_paths)

# For quick testing


def get_sample_file():
    file_path = Path("Data/genres_original/hiphop/hiphop.00052.wav")
    audio, genre = read_audio_and_genre(file_path)
    return (audio[:, :660000], genre)


def get_dataloader_from_split(split_path):
    audios, genres = load_split(split_path)
    label_ints = [label2int[x] for x in genres]
    return DataLoader(list(zip(audios, label_ints)), batch_size=32)
