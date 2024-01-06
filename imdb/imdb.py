import os
import glob
import random
import shutil
import re
from math import ceil, floor
from datasets import Dataset


def download_dataset(ds_dir, forced_extract=False, n_valid=2000):
        assert n_valid % 2 == 0, "The number of validation samples should be even."

        # Download dataset
        os.system("wget -q -nc https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")

        # Extract and create validation set
        if forced_extract or not os.path.exists(ds_dir):
            os.system(f"rm -rf {ds_dir}")
            os.system(f"tar -xf aclImdb_v1.tar.gz --one-top-level={ds_dir} --strip-components=1")

            # Verify the number of files are correct
            train_pos_files = glob.glob(f"{ds_dir}/train/pos/*.txt")
            train_neg_files = glob.glob(f"{ds_dir}/train/neg/*.txt")

            # Create validation set
            os.system(f"mkdir -p {ds_dir}/val/pos {ds_dir}/val/neg")
            random.shuffle(train_pos_files)
            random.shuffle(train_neg_files)
            val_pos_files = train_pos_files[:n_valid//2]
            val_neg_files = train_neg_files[:n_valid//2]
            for f in val_pos_files: shutil.move(f, f"{ds_dir}/val/pos")
            for f in val_neg_files: shutil.move(f, f"{ds_dir}/val/neg")


def create_samples(files, sentiment):
    data = []
    REPLACE_NO_SPACE = re.compile("[.;:!'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    for path in files:
        with open(path, "r") as f:
            text = f.read()
        sentence = REPLACE_NO_SPACE.sub("", text.strip())
        sentence = REPLACE_WITH_SPACE.sub("", sentence)
        sentence = sentence.strip()
        data.append({"sentence": sentence, "label": int(sentiment == "positive")})
    return data


def create_dataset(ds_dir="imdb_ds", ds_split="train", limit=None):
    assert ds_split in ["train", "val", "test"], "Invalid dataset split"
    pos_dir = os.path.join(ds_dir, ds_split, "pos")
    neg_dir = os.path.join(ds_dir, ds_split, "neg")
    pos_files = glob.glob(f"{pos_dir}/*.txt")
    neg_files = glob.glob(f"{neg_dir}/*.txt")

    # Limit the number of samples
    if limit is not None:
        random.shuffle(pos_files)
        random.shuffle(neg_files)
        pos_files = pos_files[:floor(limit/2)]
        neg_files = neg_files[:ceil(limit/2)]
    
    data = create_samples(pos_files, "positive")
    data.extend(create_samples(neg_files, "negative"))
    hf_ds = Dataset.from_list(data)
    return hf_ds


if __name__ == '__main__':
    ds_dir = 'imdb_ds'
    download_dataset(ds_dir)
    train_ds = create_dataset(ds_dir=ds_dir, limit=16)
    print(train_ds[0])
