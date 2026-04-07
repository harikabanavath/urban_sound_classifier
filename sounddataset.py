from torch.utils.data import Dataset
import torch
import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class sound_dataset(Dataset):
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.device = device
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    #length of sound_dataset
    def __len__(self):
        return len(self.annotations)

    #sound_dataset[index] = sound_dataset.__getitem__(index)
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)

        signal, sr = librosa.load(audio_sample_path, sr=self.target_sample_rate)
        signal = torch.from_numpy(signal).float().to(self.device)
        signal = signal.unsqueeze(0)

        signal = self._mix_down(signal)
        signal = self._right_pad(signal)
        signal = self._cut(signal)

        signal = self.transformation(signal)

        return signal, label

    #if more than 1 channels
    def _mix_down(self, signal):
        if signal.shape[0] > 1 and len(signal.shape) > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        return signal

    #if less than sample size, need to add zeroes at the right end
    def _right_pad(self, signal):
        if signal.shape[1] < self.num_samples:
            missing = self.num_samples - signal.shape[1]
            padding = (0, missing)
            signal = torch.nn.functional.pad(signal, padding)

        return signal

    #if longer than necessary, need to trim it down to sample size
    def _cut(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]

        return signal

    def _get_audio_sample_path(self, index):
        folder = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, folder, self.annotations.iloc[index, 0])

        return path

    #which folder does the audio file exist in
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]

#replica of mel spectrogram class from torch audio
class MelSpectrogram:
    def __init__(self,
                 sample_rate,
                 n_fft=1024,
                 hop_length=512,
                 n_mels=64):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, signal):
        signal_np = signal.squeeze(0).cpu().numpy()

        mel_spectrogram = librosa.feature.melspectrogram(
            y=signal_np,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        mel_spectrogram_tensor = torch.from_numpy(mel_spectrogram_db).float().unsqueeze(0)

        return mel_spectrogram_tensor

def make_train_test_split(annotations_file, test_size=0.2, random_state=42):
    df = pd.read_csv(annotations_file)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['class']
    )

    train_df.to_csv('train_sound.csv', index=False)
    test_df.to_csv('test_sound.csv', index=False)

    return train_df, test_df

if __name__ == "__main__":
    ANNOTATIONS_FILE = r"C:\Users\LALITHA\Desktop\sound_data\UrbanSound8K.csv"
    AUDIO_DIR = r"C:\Users\LALITHA\Desktop\sound_data\audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"device - {device}")

    mel_spectrogram = MelSpectrogram(
        sample_rate = SAMPLE_RATE,
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

    #print(f"no. of audio samples = {len(sd)}")

    #signal, label = sd[0]
    #print(f"signal shape - {signal.shape}")
    #print(f"label - {label}")

    make_train_test_split(ANNOTATIONS_FILE)