import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

from sounddataset import sound_dataset, MelSpectrogram
from cnn import CNN_network

BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = r"C:\Users\LALITHA\Desktop\sound_data\UrbanSound8K.csv"
AUDIO_DIR = r"C:\Users\LALITHA\Desktop\sound_data\audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loader(train_data, batch_size):
    train_data_loader = DataLoader(train_data, batch_size=batch_size)

    return train_data_loader

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        #calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        #backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"loss - {loss.item()}")

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("---------------------")
    print("training done")

def get_class_counts(annotations_file):
    df = pd.read_csv(annotations_file)
    class_counts = [0] * 10
    for i in range(len(df)):
        class_label = df.iloc[i, 6]
        class_counts[class_label] += 1
    return class_counts

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"device - {device}")

    mel_spectrogram = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    sd = sound_dataset(ANNOTATIONS_FILE,
                       AUDIO_DIR,
                       mel_spectrogram,
                       SAMPLE_RATE,
                       NUM_SAMPLES,
                       device)

    train_data_loader = create_data_loader(sd, BATCH_SIZE)

    cnn = CNN_network().to(device)

    #loss function + optimizer
    class_counts = get_class_counts(ANNOTATIONS_FILE)
    print(f"class distribution: {class_counts}")

    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    class_weights_tensor = torch.tensor(class_weights).float().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(cnn.parameters(),
                                 lr = LEARNING_RATE)

    #train the model
    train(cnn, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    #save the model
    torch.save(cnn.state_dict(), "cnn.pth")

    print("model trained")