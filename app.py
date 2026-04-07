import streamlit as st
import torch
import librosa
import soundfile as sf
from pydub import AudioSegment
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

from cnn import CNN_network
from sounddataset import sound_dataset, MelSpectrogram
from train_cnn import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

#css for styling
st.markdown("""
    <style>  
        .main-header, h1 {
            font-size: 3rem; 
            color: #1f77b4; 
            text-align: center; 
            margin-bottom: 2rem; 
        } 
        .confidence-bar {
            height: 20px; 
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4); 
            border-radius: 10px; 
            margin: 5px 0; 
        }  
        
        [data-testid="stSidebar"] {
            background-color: #cfe1c9; 
        } 
        
        [data-testid="stSidebar"] * {
            color: #545454 !important; 
        } 
        
    </style>
    """, unsafe_allow_html=True)

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

def load_model():
    try:
        cnn = CNN_network()
        state_dict = torch.load("cnn.pth", map_location=torch.device('cpu'))
        cnn.load_state_dict(state_dict)
        cnn.eval()
        return cnn
    except Exception as e:
        st.error(f"error in loading model - {e}")
        return None

def preprocess_audio(audio_file):
    try:
        audio_file.seek(0)

        try:
            waveform, sample_rate = librosa.load(io.BytesIO(audio_file.read()), sr=SAMPLE_RATE)
            audio_file.seek(0)
        except:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    audio_file.seek(0)
                    tmp_file.write(audio_file.read())
                    tmp_file.flush()
                    waveform, sample_rate = sf.read(tmp_file.name)
                    os.unlink(tmp_file.name)
                if sample_rate != SAMPLE_RATE:
                    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            except:
                try:
                    audio_file.seek(0)
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_file.read()))
                    audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1)

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        audio_segment.export(tmp_file.name, format="wav")
                        waveform, sample_rate = sf.read(tmp_file.name)
                        os.unlink(tmp_file.name)
                except Exception as e:
                    st.error(f"error in processing audio - {e}")
                    return None

        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=0)

        if len(waveform) > NUM_SAMPLES:
            waveform = waveform[:NUM_SAMPLES]
        else:
            padding = NUM_SAMPLES - len(waveform)
            waveform = np.pad(waveform, (0, padding))

        waveform_tensor = torch.from_numpy(waveform).float()
        waveform_tensor = waveform_tensor.unsqueeze(0)

        mel_spectrogram = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        spectrogram = mel_spectrogram(waveform_tensor)

        return spectrogram.unsqueeze(0)

    except Exception as e:
        st.error(f"error in processing audio - {e}")
        return None

def predict(model, input_tensor, class_mapping):
    with torch.no_grad():
        predictions = model(input_tensor)
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        predicted_index = predictions[0].argmax(0)
        predicted_class = class_mapping[predicted_index]

        class_probs = {class_mapping[i]: float(prob) for i, prob in enumerate(probabilities)}

    return predicted_class, class_probs

def create_audio_visualization(audio_file):
    try:
        if isinstance(audio_file, str):
            audio_data, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
        else:
            audio_file.seek(0)
            audio_data, sr = librosa.load(io.BytesIO(audio_file.read()), sr=SAMPLE_RATE)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(audio_data, color='#1f77b4')
        ax1.set_title('AUDIO WAVEFORM', fontsize=14, fontweight='bold')
        ax1.set_ylabel('amplitude')
        ax1.grid(True, alpha=0.3)

        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='viridis')
        ax2.set_title('SPECTROGRAM', fontsize=14, fontweight='bold')
        ax2.set_xlabel('time (in sec)')
        ax2.set_ylabel('frequency (in Hz)')
        plt.colorbar(img, ax=ax2, format='%+2.0f dB')

        plt.tight_layout()

        plt.subplots_adjust(hspace=0.8)
        return fig

    except Exception as e:
        st.error(f"error in visualizations - {e}")
        return None

def main():
    st.markdown('<h1 class="main-header">URBAN SOUND CLASSIFIER</h1>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error('failed to load model')
        return

    with st.sidebar:
        st.header("ABOUT")
        st.markdown("""
            classifies urban sounds into 10 categories  
            - air conditioner 
            - car horn 
            - children playing 
            - dog bark 
            - drilling 
            - engine idling 
            - gun shot 
            - jackhammer 
            - siren 
            - street music  
        """)

        st.header("MODEL INFO")
        st.markdown(f"""
                    - dataset - UrbanSound8K 
                    - model - CNN trained on mel spectrograms 
                    - sample rate - {SAMPLE_RATE} 
                    - target samples - {NUM_SAMPLES}                    
                """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "upload your audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help='upload wav, mp3, flac, or m4a files'
        )

        if uploaded_file is not None:
            st.success(f"file uploaded - {uploaded_file.name}")

            st.audio(uploaded_file, format='audio/wav')

    with col2:
        if uploaded_file is not None:
            with st.spinner("analyzing audio "):
                input_tensor = preprocess_audio(uploaded_file)

                if input_tensor is not None:
                    predicted_class, class_probs = predict(model, input_tensor, class_mapping)

                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.subheader("PREDICTION RESULT")

                    st.success(f"**predicted sound:** {predicted_class.replace('_', ' ')}")

                    sorted_probs = sorted(class_probs.items(), key=lambda x:x[1], reverse=True)

                    for class_name, prob in sorted_probs:
                        display_name = class_name.replace("_", " ")
                        col_a, col_b, col_c = st.columns([2, 3, 1])

                        with col_a:
                            st.write(f"**{display_name}**")

                        with col_b:
                            progress_html = f"""
                            <div style="width: 100%; background-color: #ddd; border-radius: 10px;">
                                <div style="width: {prob * 100}%; height: 20px; background: linear-gradient(90deg, #ff6b6b, #4ecdc4); border-radius: 10px; text-align: center; color: white; line-height: 20px;">
                                </div>
                            </div>
                            """
                            st.markdown(progress_html, unsafe_allow_html=True)

                        with col_c:
                            st.write(f"{prob*100:.1f}%")

                    st.markdown('</div>', unsafe_allow_html=True)


                with st.expander("test with sample sounds"):
                    st.markdown("""
                        don't have an audio file that works? 
                        this model was trained on the UrbanSound8K dataset 
                        
                        you can find sample sounds from 
                        - [UrbanSound8K Dataset] (https://urbansounddataset.weebly.com/) 
                        - [Freesound.org] (https://freesound.org/)  
                    """)

    if uploaded_file is not None:
        st.subheader("AUDIO VISUALIZATION")
        viz_fig = create_audio_visualization(uploaded_file)
        if viz_fig:
            st.pyplot(viz_fig)

if __name__ == "__main__":
    main()