import tensorflow as tf
from tensorflow.keras import losses, optimizers, preprocessing, applications
import tensorflow_text as tf_text
from tensorflow.lite.python import interpreter
import numpy as np
from models import Transformer, Captioner
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import time
import os
from tqdm import tqdm


""" helpers """

#displays time as h:mm:ss
def format_time(seconds):
    return "{}:{:0>2}:{:0>2}".format(int(seconds//3600), int((seconds//60)%60), int(seconds%60))


""" processing the dataset """

target_vocab = open("dataset/vocab.txt", encoding="utf-8").read().splitlines()
target_tokenizer = tf_text.FastWordpieceTokenizer(vocab=target_vocab, suffix_indicator="##", max_bytes_per_word=100, token_out_type=tf.int32,
                                                              unknown_token='<UNK>', no_pretokenization=True, support_detokenization=True, model_buffer=None)

inputs = np.load("dataset/features.npz")["features"]
TRAIN_SIZE = 40000
BATCH_SIZE = 16
train_inputs = inputs[:TRAIN_SIZE]
train_inputs = np.reshape(train_inputs, [-1, BATCH_SIZE, train_inputs.shape[-2], train_inputs.shape[-1]])
train_targets = np.load("dataset/captions.npz")["captions"][:TRAIN_SIZE]
train_targets = np.reshape(train_targets, [-1, BATCH_SIZE, train_targets.shape[-1]])

test_inputs = inputs[TRAIN_SIZE:]
test_images = pd.read_csv("dataset/captions.csv")["image"][TRAIN_SIZE:].values
test_targets = pd.read_csv("dataset/captions.csv")["caption"][TRAIN_SIZE:].values

# sanity check
print(f"train_inputs.shape: {train_inputs.shape}")
print(f"train_targets.shape: {train_targets.shape}")
print(f"test_images.shape: {test_images.shape}")
print(f"test_inputs.shape: {test_inputs.shape}")
print(f"test_targets.shape: {test_targets.shape}")
print(f"vocab size: {len(target_vocab)}")

def get_batch(batch_size):
    """
        Gets a batch from test data. Meant only for inference.
    """
    assert batch_size < len(test_images)
    indices = np.random.choice(len(test_images), batch_size, replace=False)
    filenames = test_images[indices]
    img_batch = []
    for filename in filenames:
        img = Image.open(f"dataset/resized/{filename}")
        img = np.array(img).astype(np.float32)
        img = np.expand_dims(img, axis=0)  # add concat dimension
        img_batch.append(img)

    inp_batch = test_inputs[indices]
    img_batch = np.concatenate(img_batch, axis=0)
    tar_batch = test_targets[indices]
    return img_batch, inp_batch, tar_batch


""" model """

embedding_dim = 256
model = Transformer(num_layers=4, d_model=embedding_dim, num_heads=8, dff=800, target_vocab_size=len(target_vocab),
                    d_row=7, d_col=7, pe_target=train_targets.shape[-1], embedding_matrix=None)
# model.build(input_shape=[[None, train_inputs.shape[-2], train_inputs.shape[-1]], [None, train_targets.shape[-1]]])
model.build(input_shape=[[None, train_inputs.shape[-2], train_inputs.shape[-1]], [None, train_targets.shape[-1]]])

# model.load_weights("models/weights.h5")
#
#
# captioner = Captioner(model, max_length=train_targets.shape[-1])
# inp, _, _ = get_batch(batch_size=1)
# out = captioner(inp[0])
# print(out)
# tokens = open("alt_vocab.txt", "r", encoding="utf-8").read().splitlines()
# out = "".join([tokens[i] for i in out])
# print(out)


""" training configuration """

loss_object = losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

class CustomSchedule(optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(embedding_dim)
optimizer = optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
    pred = tf.argmax(pred, axis=2)
    pred = tf.cast(pred, dtype=real.dtype)
    accuracies = tf.equal(real, pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


""" training """

@tf.function(input_signature=[
    tf.TensorSpec(shape=(None, None, None), dtype=tf.int64),  # (batch_size, seq_len)
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
])
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    with tf.GradientTape() as tape:
        preds, _ = model([inp, tar_inp], training = True)
        loss = loss_function(tar_real, preds)
        accuracy = accuracy_function(tar_real, preds)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, accuracy


def train(num_epochs=50):
    loss_history = []
    accuracy_history = []
    prev_time = time.time()
    time_elapsed = 0

    # load saved models
    if os.path.isfile("models/weights.h5"):
        model.load_weights("models/weights.h5")

    print("Training...")

    for epoch in range(num_epochs):
        for inp, tar in tqdm(zip(train_inputs, train_targets)):
            loss, accuracy = train_step(inp, tar)
            loss_history.append(loss.numpy().mean())
            accuracy_history.append(accuracy.numpy().mean())

            time_elapsed += time.time() - prev_time
            prev_time = time.time()

        print(f"Epoch {epoch + 1}/{num_epochs}. Loss: {loss_history[-1]}. Accuracy: {accuracy_history[-1]}. Time elapsed: {format_time(time_elapsed)}\n")
        # save checkpoints
        model.save_weights("models/weights.h5")
        model.save_weights(f"models/epoch{epoch + 1}.h5")

        # plot a graph that will show how our loss varied with time
        plt.plot(loss_history)
        plt.plot(accuracy_history)
        plt.title("Training Progress")
        plt.xlabel("Iterations")
        plt.legend(["Loss", "Accuracy"])
        plt.savefig(os.path.join("./plots/TrainingProgress"))
        # plt.show()
        plt.close()

        imgs, _, captions = get_batch(5)
        for idx, (img, caption) in enumerate(zip(imgs, captions)):
            os.makedirs(f"generated/epoch {epoch+1}", exist_ok=True)
            Image.fromarray(img.astype(np.uint8)).save(f"generated/epoch {epoch+1}/img{idx}.jpg")
            generated = generate(img)
            open(f"generated/epoch {epoch+1}/predicted{idx}.txt", "w", encoding="utf-8").write(generated)
            open(f"generated/epoch {epoch+1}/true{idx}.txt", "w", encoding="utf-8").write(caption)


""" inference """

def generate(img, max_length = train_targets.shape[-1]):
    # initialize start token
    target = np.array([2]) # 2 - <BOS>

    img_model = applications.EfficientNetB0(include_top=False, weights='imagenet')    # we discard the last layer because we only want the features not the classes
    img = np.expand_dims(img, 0)
    img = applications.efficientnet.preprocess_input(img)
    features = img_model(img)
    features = np.reshape(features, [features.shape[0], -1, features.shape[-1]])  # None,7,7,960 -> None,49,960

    for i in range(max_length):
        prediction, _ = model([features, np.expand_dims(target, 0)], training=False)
        prediction = prediction[0, -1, :]  # we only need the last timestep to append it to the target. shape: [batch_size, vocab_size]
        # sample the distribution
        prediction = tf.math.top_k(prediction, k=2).indices  # shape: [2]
        prediction = tf.where(tf.not_equal(prediction[0], 1), prediction[0], prediction[1])  # sample the most probable token after <UNK> (1), if <UNK> is the predicted token
                                                                                             # we don't want <UNK> in our sequence
        if prediction == 3: # 3 - <EOS>
            break

        prediction = np.expand_dims(prediction, 0)
        target = np.append(target, prediction)

    target = target[1:]  # remove start token
    target = target_tokenizer.detokenize(target).numpy().decode("utf-8")
    target = target.replace(" <UNK>", "").replace(" <PAD>", "").replace(" <BOS>", "").replace(" <EOS>", "")
    return target

def generate_from_saved_weights(num_samples=5):
    for weights in os.listdir("./models"):
        model.load_weights(f"./models/{weights}")
        imgs, _, captions = get_batch(num_samples)
        for idx, (img, caption) in enumerate(zip(imgs, captions)):
            os.makedirs(f"generated/{weights}-{idx}", exist_ok=True)
            Image.fromarray(img.astype(np.uint8)).save(f"generated/{weights}-{idx}/img.jpg")
            generated = generate(img)
            open(f"generated/{weights}-{idx}/predicted.txt", "w", encoding="utf-8").write(generated)
            open(f"generated/{weights}-{idx}/true.txt", "w", encoding="utf-8").write(caption)


""" deployment"""

def create_tflite_model():
    captioner = Captioner(model, max_length=train_targets.shape[-1])
    converter = tf.lite.TFLiteConverter.from_keras_model(captioner)
    output = converter.convert()
    open("captioner.tflite", "wb").write(output)

def run_tflite_model(image):
    tflite_model = open("captioner.tflite", "rb").read()
    interp = interpreter.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    print(input_details)
    output_details = interp.get_output_details()
    print(output_details)
    interp.set_tensor(input_details[0]['index'], image)
    interp.invoke()
    output_data = interp.get_tensor(output_details[0]['index'])
    return output_data


if __name__ == "__main__":
    model.load_weights("models/weights.h5")
    # train()
    # generate_from_saved_weights(100)


    # imgs, _, captions = get_batch(5)
    # for idx, (img, caption) in enumerate(zip(imgs, captions)):
    #     os.makedirs(f"generated/{idx}", exist_ok=True)
    #     Image.fromarray(img.astype(np.uint8)).save(f"generated/{idx}/img.jpg")
    #     generated = generate(img)
    #     open(f"generated/{idx}/true.txt", "w", encoding="utf-8").write(caption)
    #     open(f"generated/{idx}/predicted.txt", "w", encoding="utf-8").write(generated)



    # create_tflite_model()


    img, _, _ = get_batch(batch_size=1)
    out = run_tflite_model(img[0])
    print(out)
    tokens = open("alt_vocab.txt", "r", encoding="utf-8").read().splitlines()
    out = "".join([tokens[i] for i in out])
    print(out)

    pass
