import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 📂 مسیر فایل‌های صوتی
audio_folder = "Data/processed"
output_csv = "Data/features/spectral_centroid.csv"

# 🎵 تابع استخراج Spectral Centroid
def extract_spectral_centroid(file_path, sr=22050):
    try:
        # 🔹 1. بارگذاری فایل صوتی
        y, sr = librosa.load(file_path, sr=sr)

        # 🔹 2. پردازش اولیه (نرمال‌سازی)
        y = librosa.util.normalize(y)

        # 🔹 3. محاسبه Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        return spectral_centroid
    except Exception as e:
        print(f"❌ خطا در پردازش {file_path}: {e}")
        return None

# 📝 پردازش فایل‌ها و استخراج ویژگی‌ها
data = []
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)

    # استخراج Spectral Centroid
    spectral_centroid = extract_spectral_centroid(file_path)
    if spectral_centroid is not None:
        # استخراج شماره دانشجویی از نام فایل
        parts = file.split('_')
        if len(parts) >= 4:  # Ensure correct filename format
            student_id = parts[2]  # شماره دانشجویی
            gender = parts[3].split('.')[0]  # جنسیت (male/female)
            
            # محاسبه میانگین ویژگی‌ها
            centroid_features = np.mean(spectral_centroid, axis=1)  # میانگین‌گیری در طول زمان

            # ذخیره اطلاعات در لیست
            feature_dict = {"filename": file, "student_id": student_id}
            for i in range(len(centroid_features)):
                feature_dict[f'centroid_{i+1}'] = centroid_features[i]

            data.append(feature_dict)

# 📊 تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# 📁 ذخیره در فایل CSV
df.to_csv(output_csv, index=False)
print(f"✅ ویژگی‌های Spectral Centroid در '{output_csv}' ذخیره شد.")

# Plot Spectral Centroid for a specific example
file_path = "Data/processed/HW1_intro_610300032_male_segment_1.wav"
spectral_centroid = extract_spectral_centroid(file_path)
plt.figure(figsize=(10, 5))
librosa.display.specshow(spectral_centroid, x_axis='time', sr=22050, cmap='coolwarm')
plt.colorbar(label="Spectral Centroid")
plt.title("Spectral Centroid")
plt.xlabel("Time")
plt.ylabel("Frequency Bands")
plt.show()
