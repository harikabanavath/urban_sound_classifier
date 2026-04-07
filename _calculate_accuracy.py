import torch
from torch.utils.data import DataLoader
from cnn import CNN_network
from sounddataset import sound_dataset, MelSpectrogram
from train_cnn import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES


def calculate_test_accuracy(model, test_loader, device):
    """Calculate accuracy on test set"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def calculate_class_wise_accuracy(model, test_loader, device, class_mapping):
    """Calculate accuracy for each class"""
    model.eval()
    class_correct = [0] * len(class_mapping)
    class_total = [0] * len(class_mapping)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    class_accuracy = {}
    for i in range(len(class_mapping)):
        if class_total[i] > 0:
            class_accuracy[class_mapping[i]] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracy[class_mapping[i]] = 0

    return class_accuracy


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    cnn = CNN_network().to(device)
    state_dict = torch.load("cnn.pth", map_location=device)
    cnn.load_state_dict(state_dict)

    # Prepare test dataset (you might need to split your data)
    mel_spectrogram = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # Assuming you have a way to get test data
    # You might need to modify UrbanSoundDataset to support train/test split
    test_dataset = sound_dataset(
        ANNOTATIONS_FILE,  # You might want a separate test annotations file
        AUDIO_DIR,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device.type
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    class_mapping = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
    ]

    # Calculate overall accuracy
    accuracy = calculate_test_accuracy(cnn, test_loader, device)
    print(f"Overall Test Accuracy: {accuracy:.2f}%")

    # Calculate class-wise accuracy
    class_accuracy = calculate_class_wise_accuracy(cnn, test_loader, device, class_mapping)
    print("\nClass-wise Accuracy:")
    for class_name, acc in class_accuracy.items():
        print(f"{class_name}: {acc:.2f}%")