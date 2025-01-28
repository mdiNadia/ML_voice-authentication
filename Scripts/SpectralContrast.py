#Spectral Contrast
# یک ویژگی صوتی پرکاربرد در پردازش گفتار و موسیقی است که 
# تفاوت بین پیک‌های فرکانسی 
# (high energy) و
# فرکانس‌های پایین‌تر 
# (low energy)
# را اندازه‌گیری می‌کند

import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 📂 مسیر فایل‌های صوتی
audio_folder = "Data/processed"
output_csv = "Data/features/spectral_contrast.csv"

# 🎵 تابع استخراج Spectral Contrast
def extract_spectral_contrast(file_path, sr=22050, n_bands=6):
    try:
        # 🔹 1. بارگذاری فایل صوتی
        y, sr = librosa.load(file_path, sr=sr)

        # 🔹 2. پردازش اولیه (نرمال‌سازی)
        y = librosa.util.normalize(y)

        # 🔹 3. محاسبه STFT (تبدیل فوریه کوتاه-مدت)
        stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))

        # 🔹 4. محاسبه Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=n_bands)

        return spectral_contrast
    except Exception as e:
        print(f"❌ خطا در پردازش {file_path}: {e}")
        return None

# 📝 پردازش فایل‌ها و استخراج ویژگی‌ها
data = []
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)

    # استخراج Spectral Contrast
    spectral_contrast = extract_spectral_contrast(file_path)
    if spectral_contrast is not None:
        # استخراج شماره دانشجویی از نام فایل
        parts = file.split('_')
        student_id = parts[2]  # شماره دانشجویی
        gender = parts[3].split('.')[0]  # جنسیت (male/female)
        # محاسبه میانگین ویژگی‌ها
        contrast_features = np.mean(spectral_contrast, axis=1)  # میانگین‌گیری در طول زمان

        # ذخیره اطلاعات در لیست
        feature_dict = {"filename": file, "student_id": student_id}
        for i in range(len(contrast_features)):
            feature_dict[f'contrast_{i+1}'] = contrast_features[i]

        data.append(feature_dict)

# 📊 تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# 📁 ذخیره در فایل CSV
df.to_csv(output_csv, index=False)
print(f"✅ ویژگی‌های Spectral Contrast در '{output_csv}' ذخیره شد.")


file_path = "Data/processed/HW1_intro_610300032_male_segment_1.wav"
spectral_contrast = extract_spectral_contrast(file_path)
plt.figure(figsize=(10, 5))
librosa.display.specshow(spectral_contrast, x_axis='time', sr=22050, cmap='coolwarm')
plt.colorbar(label="Contrast")
plt.title("Spectral Contrast")
plt.xlabel("Time")
plt.ylabel("Frequency Bands")
plt.show()


