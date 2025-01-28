import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy import signal
from pyloudnorm import Meter, normalize

# مشخص کردن پوشه ورودی و خروجی
input_directory = './Data/raw'
output_directory = './Data/processed'
output_csv = os.path.join(output_directory, 'audio_segments.csv')

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
    if len(frames) > 0:
        return np.concatenate([audio_data[i * hop_length:(i + 1) * hop_length] for i in frames])
    else:
        return np.array([])

# نرمال‌سازی صوت
def normalize_audio(audio_data, sr, target_lufs=-14):
    meter = Meter(sr)
    loudness = meter.integrated_loudness(audio_data)
    return normalize.loudness(audio_data, loudness, target_lufs)

# تبدیل لیبل به عدد
def label_to_number(label):
    if "_male.mp3" in label.lower() or "-male.mp3" in label.lower():
        return 1
    elif "_female.mp3" in label.lower() or "-female.mp3" in label.lower():
        return 0
    return -1  # در صورتی که لیبل شناخته شده نباشد

# پردازش یک فایل
def process_file(input_file, output_directory, output_data, lowcut=50.0, highcut=5000.0, sr=22050, segment_duration=3):
    audio, sr = librosa.load(input_file, sr=sr)
    filtered_audio = bandpass_filter(audio, sr, lowcut, highcut)
    non_silent_audio = remove_silence(filtered_audio, sr, threshold=0.05)
    if non_silent_audio.size > 0:
        normalized_audio = normalize_audio(non_silent_audio, sr, target_lufs=-14)
        label = label_to_number(os.path.basename(input_file))
        num_samples_per_segment = sr * segment_duration
        total_segments = len(normalized_audio) // num_samples_per_segment

        for i in range(total_segments):
            start = i * num_samples_per_segment
            end = start + num_samples_per_segment
            segment = normalized_audio[start:end]
            segment_file = os.path.join(output_directory, f"{os.path.basename(input_file).split('.')[0]}_segment_{i}.wav")
            sf.write(segment_file, segment, sr)
            output_data.append([label, segment_file])

# پردازش فایل‌های پوشه
def process_directory(input_directory, output_directory, output_csv, lowcut=50.0, highcut=5000.0, sr=22050, segment_duration=3):
    audio_files = glob.glob(os.path.join(input_directory, "*.mp3")) + glob.glob(os.path.join(input_directory, "*.wav"))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_data = []
    
    for audio_file in audio_files:
        process_file(audio_file, output_directory, output_data, lowcut, highcut, sr, segment_duration)
    
    df = pd.DataFrame(output_data, columns=['label', 'segment_file'])
    df.to_csv(output_csv, index=False)
    print(f"Saved segments to {output_csv}")

# اجرا
process_directory(input_directory, output_directory, output_csv)
