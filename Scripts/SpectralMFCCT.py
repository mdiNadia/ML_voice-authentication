import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

audio_folder = "Data/processed"
output_csv = "Data/features/mfcc_features.csv"

def extract_mfcc(file_path, n_mfcc=13):
    try:
        # 🔹 1. بارگذاری فایل صوتی
        y, sr = librosa.load(file_path, sr=22050)

        # 🔹 2. پردازش اولیه (نرمال‌سازی)
        y = librosa.util.normalize(y)

        # 🔹 3. تبدیل سیگنال به STFT
        stft = librosa.stft(y, n_fft=512, hop_length=256, win_length=400, window="hann")

        # 🔹 4. محاسبه طیف توان (Power Spectrogram)
        power_spectrogram = np.abs(stft) ** 2

        # 🔹 5. اعمال فیلتر Mel
        mel_filterbank = librosa.filters.mel(sr=sr, n_fft=512, n_mels=40)
        mel_spectrogram = np.dot(mel_filterbank, power_spectrogram)

        # 🔹 6. تبدیل به مقیاس لگاریتمی
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        # 🔹 7. اعمال DCT (برای حذف همبستگی)
        mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sr, n_mfcc=n_mfcc)

        # 🔹 8. انتخاب 13 ویژگی برتر
        mfcc_features = np.mean(mfcc, axis=1)  # میانگین‌گیری روی بعد زمان

        return mfcc_features
    except Exception as e:
        print(f"❌ خطا در پردازش {file_path}: {e}")
        return None

# 📝 استخراج ویژگی‌های MFCC از تمام فایل‌های پوشه
data = []
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)
    
    # استخراج ویژگی MFCC
    mfcc_features = extract_mfcc(file_path)
    if mfcc_features is not None:
        # استخراج شماره دانشجویی و جنسیت از نام فایل
        parts = file.split('_')
        student_id = parts[2]  # شماره دانشجویی
        gender = parts[3].split('.')[0]  # جنسیت (male/female)

        # ایجاد دیکشنری برای ذخیره داده‌ها
        feature_dict = {"filename": file, "student_id": student_id, "gender": gender}
        for i in range(len(mfcc_features)):
            feature_dict[f'mfcc_{i+1}'] = mfcc_features[i]
        
        data.append(feature_dict)

# 📊 تبدیل داده‌ها به دیتافریم
df = pd.DataFrame(data)

# 📁 ذخیره در فایل CSV
df.to_csv(output_csv, index=False)
print(f"✅ ویژگی‌های MFCC در فایل '{output_csv}' ذخیره شد.")


file_path = "Data/processed/HW1_intro_610300032_male_segment_1.wav"
mfcc = extract_mfcc(file_path)

# Check if MFCC is not None and has valid shape
if mfcc is not None and mfcc.shape[1] > 0:
    # Plot the MFCC
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time", sr=22050)
    plt.colorbar(label="MFCC")
    plt.title("Mel Frequency Cepstral Coefficients")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.show()
else:
    print("MFCC extraction failed or has invalid shape.")

