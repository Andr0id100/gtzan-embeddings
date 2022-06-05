from fastcore.xtras import Path
from data_utils import get_file_names
import random
from sklearn.model_selection import train_test_split
import os

SEED = 23

random.seed(SEED)

data_root = Path("Data/genres_original/")
file_paths = get_file_names(data_root)

# Removing a faulty file
file_paths.pop(file_paths.index(Path("Data/genres_original/jazz/jazz.00054.wav")))

random.shuffle(file_paths)

embedding_train_split, hold_out_split = train_test_split(file_paths, test_size=0.4, random_state=SEED)
embedding_test_split, classifier_test_split = train_test_split(hold_out_split, test_size=0.5, random_state=SEED)

print("Embedding Train:", len(embedding_train_split))
print("Embedding Test:", len(embedding_test_split))
print("Classifier Test", len(classifier_test_split))

def write_split_file(split_file_paths, file_name):
    os.makedirs("splits", exist_ok=True)
    with open(f"splits/{file_name}", 'w') as f:
        for file_path in split_file_paths:
            f.write(f"{str(file_path)}\n")


write_split_file(embedding_train_split, "embedding_train_split.txt")
write_split_file(embedding_test_split, "embedding_test_split.txt")
write_split_file(classifier_test_split, "classifier_test_split.txt")