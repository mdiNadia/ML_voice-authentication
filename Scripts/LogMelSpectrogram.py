import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 📂 مسیر فایل‌های صوتی
audio_folder = "Data/processed"
output_csv = "Data/features/log_mel_spectrogram.csv"

# 🎵 تابع استخراج Log Mel Spectrogram
def extract_log_mel_spectrogram(file_path, sr=22050, n_mels=40):
    try:
        # 🔹 1. بارگذاری فایل صوتی
        y, sr = librosa.load(file_path, sr=sr)

        # 🔹 2. پردازش اولیه (نرمال‌سازی)
        y = librosa.util.normalize(y)

        # 🔹 3. محاسبه Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=512, hop_length=256)

        # 🔹 4. تبدیل به مقیاس لاگاریتمی (Log Mel Spectrogram)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        return log_mel_spectrogram
    except Exception as e:
        print(f"❌ خطا در پردازش {file_path}: {e}")
        return None

# 📝 پردازش فایل‌های صوتی و استخراج ویژگی‌ها
data = []
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)
    
    # استخراج Log Mel Spectrogram
    log_mel_spectrogram = extract_log_mel_spectrogram(file_path)
    if log_mel_spectrogram is not None:
        # استخراج شماره دانشجویی از نام فایل
        parts = file.split('_')
        student_id = parts[2]  # شماره دانشجویی
        gender = parts[3].split('.')[0]  # جنسیت (male/female)
        
        # محاسبه میانگین طیف Log Mel
        log_mel_features = np.mean(log_mel_spectrogram, axis=1)  # میانگین‌گیری در طول زمان

        # ذخیره اطلاعات در لیست
        feature_dict = {"filename": file, "student_id": student_id, "gender": gender}
        for i in range(len(log_mel_features)):
            feature_dict[f'log_mel_{i+1}'] = log_mel_features[i]

        data.append(feature_dict)

# 📊 تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# 📁 ذخیره در فایل CSV
df.to_csv(output_csv, index=False)
print(f"✅ ویژگی‌های Log Mel Spectrogram در '{output_csv}' ذخیره شد.")

# 🎨 تجسم Log Mel Spectrogram برای یک فایل نمونه
file_path = "Data/processed/HW1_intro_610300032_male_segment_1.wav"
log_mel_spectrogram = extract_log_mel_spectrogram(file_path)

plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="mel", sr=22050, cmap="coolwarm")
plt.colorbar(format="%+2.0f dB")
plt.title("Log Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.show()
