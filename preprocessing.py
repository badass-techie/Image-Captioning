import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as tf_text
from PIL import Image
from tensorflow.keras import preprocessing, applications
from tqdm import tqdm


def resample_images(path = "dataset/images/", path_to = "dataset/resized/", size=(224, 224)):
    print("Resampling images...")
    os.makedirs(path_to, exist_ok=True)
    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".jpg"):
            img = Image.open(f"{path}/{filename}")
            img = img.resize(size)
            img.save(f"{path_to}/{filename}")


csv = pd.read_csv("dataset/raw captions.csv")


def preprocess_captions():
    global csv

    # to lowercase
    csv["caption"] = csv["caption"].str.lower()
    # remove punctuation except for comma, period, and apostrophe
    csv["caption"] = csv["caption"].apply(lambda x: re.sub(r'[^a-zA-Z0-9 .,\']', '', x))
    # remove words with numbers
    csv["caption"] = csv["caption"].apply(lambda x: " ".join([word for word in x.split() if word.isalpha()]))
    # remove words with single letters
    csv["caption"] = csv["caption"].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    # add <BOS> and <EOS> to each caption
    csv["caption"] = csv["caption"].apply(lambda x: f"<BOS> {x} <EOS>")

    # save
    csv.to_csv("dataset/captions.csv", index=False)
    csv = pd.read_csv("dataset/captions.csv")


def save_vocab(num_words = 5000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<UNK>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(csv["caption"].values)

    tokenizer.word_index['<PAD>'] = 0
    tokenizer.index_word[0] = '<PAD>'

    tokenizer.word_index['<BOS>'] = 2
    tokenizer.index_word[2] = '<BOS>'

    tokenizer.word_index['<EOS>'] = 3
    tokenizer.index_word[3] = '<EOS>'

    open("dataset/vocab.txt", "w").write("\n".join(tokenizer.sequences_to_texts([range(5000)])[0].split()))



def save_tokenized_captions():
    tokens = open("dataset/vocab.txt", "r", encoding="utf-8").read().splitlines()
    tokenizer = tf_text.FastWordpieceTokenizer(vocab=tokens, suffix_indicator="##", max_bytes_per_word=100,
                                               token_out_type=tf.int32, unknown_token='<UNK>', no_pretokenization=True,
                                               support_detokenization=True, model_buffer=None)
    captions = csv["caption"].to_list()
    captions = [tokenizer.tokenize(caption.split()).numpy().flatten() for caption in captions]

    captions = preprocessing.sequence.pad_sequences(captions, padding="post")

    # save the captions
    np.savez_compressed("dataset/captions.npz", captions=captions)


def extract_image_features():
    img_model = applications.EfficientNetB0(include_top=False,
                                         weights='imagenet')  # we discard the last layer because we only want the features not the classes
    filenames = csv["image"].to_list()

    unique = {}
    features = []
    print("Extracting image features...")
    for filename in tqdm(filenames):
        if filename in unique:
            features.append(unique[filename])
        else:
            img = Image.open(f"dataset/resized/{filename}")
            img = np.array(img).astype(np.float32)
            img = np.expand_dims(img, axis=0)  # add batch dimension
            img = applications.efficientnet.preprocess_input(img)
            out = img_model(img)
            # print(out.shape)
            out = np.reshape(out, [out.shape[0], -1, out.shape[-1]])  # None,7,7,960 -> None,49,960
            features.append(out)
            unique[filename] = out

    features = np.concatenate(features, axis=0)
    np.savez_compressed("dataset/features.npz", features=features)


def create_alt_vocab():
    target_vocab = open("dataset/vocab.txt", "r", encoding="utf-8").read().splitlines()
    target_vocab = [("#" + token).replace("##", "").replace("#", " ").replace(" <UNK>", "").replace(" <PAD>", "").replace(" <BOS>", "").replace(" <EOS>", "") for token in target_vocab]
    open("alt_vocab.txt", "w", encoding="utf-8").write("\n".join(target_vocab))



if __name__ == "__main__":
    # resample_images()
    # preprocess_captions()
    # save_vocab()
    # save_tokenized_captions()
    # extract_image_features()
    # create_alt_vocab()
    pass
