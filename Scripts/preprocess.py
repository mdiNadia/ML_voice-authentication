import os
import glob
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from pyloudnorm import Meter, normalize

# مشخص کردن پوشه ورودی و خروجی
input_directory = './Data/raw'
output_directory = './Data/processed'

# طراحی فیلتر باندپاس
def bandpass_filter(audio, sr, lowcut, highcut, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return signal.filtfilt(b, a, audio)

# حذف سکوت از صوت
def remove_silence(audio_data, sr, threshold=0.1, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    frames = np.where(rms > threshold)[0]
    return np.concatenate([audio_data[i * hop_length:(i + 1) * hop_length] for i in frames])

# نرمال‌سازی صوت
def normalize_audio(audio_data, sr, target_lufs=-14):
    meter = Meter(sr)
    loudness = meter.integrated_loudness(audio_data)
    return normalize.loudness(audio_data, loudness, target_lufs)

# پردازش یک فایل
def process_file(input_file, output_file, lowcut=50.0, highcut=5000.0, sr=22050):
    audio, sr = librosa.load(input_file, sr=sr)
    filtered_audio = bandpass_filter(audio, sr, lowcut, highcut)
    non_silent_audio = remove_silence(filtered_audio, sr, threshold=0.05)
    normalized_audio = normalize_audio(non_silent_audio, sr, target_lufs=-14)
    sf.write(output_file, normalized_audio, sr)
    print(f"Processed: {input_file} -> {output_file}")

# پردازش فایل‌های پوشه
def process_directory(input_directory, output_directory, lowcut=50.0, highcut=5000.0, sr=22050):
    audio_files = glob.glob(os.path.join(input_directory, "*.mp3")) + glob.glob(os.path.join(input_directory, "*.wav"))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for audio_file in audio_files:
        output_file = os.path.join(output_directory, os.path.basename(audio_file))
        process_file(audio_file, output_file, lowcut, highcut, sr)

# اجرا
process_directory(input_directory, output_directory)
