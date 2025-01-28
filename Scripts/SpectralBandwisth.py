import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 📂 مسیر فایل‌های صوتی
audio_folder = "Data/processed"
output_csv = "Data/features/spectral_bandwidth.csv"

# 🎵 تابع استخراج Spectral Bandwidth
def extract_spectral_bandwidth(file_path, sr=22050):
    try:
        # 🔹 1. بارگذاری فایل صوتی
        y, sr = librosa.load(file_path, sr=sr)

        # 🔹 2. پردازش اولیه (نرمال‌سازی)
        y = librosa.util.normalize(y)

        # 🔹 3. محاسبه Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        return spectral_bandwidth
    except Exception as e:
        print(f"❌ خطا در پردازش {file_path}: {e}")
        return None

# 📝 پردازش فایل‌ها و استخراج ویژگی‌ها
data = []
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)

    # استخراج Spectral Bandwidth
    spectral_bandwidth = extract_spectral_bandwidth(file_path)
    if spectral_bandwidth is not None:
        # استخراج شماره دانشجویی از نام فایل
        parts = file.split('_')
        if len(parts) >= 4:  # Ensure correct filename format
            student_id = parts[2]  # شماره دانشجویی
            gender = parts[3].split('.')[0]  # جنسیت (male/female)
            
            # محاسبه میانگین ویژگی‌ها
            bandwidth_features = np.mean(spectral_bandwidth, axis=1)  # میانگین‌گیری در طول زمان

            # ذخیره اطلاعات در لیست
            feature_dict = {"filename": file, "student_id": student_id}
            for i in range(len(bandwidth_features)):
                feature_dict[f'bandwidth_{i+1}'] = bandwidth_features[i]

            data.append(feature_dict)

# 📊 تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# 📁 ذخیره در فایل CSV
df.to_csv(output_csv, index=False)
print(f"✅ ویژگی‌های Spectral Bandwidth در '{output_csv}' ذخیره شد.")

# Plot Spectral Bandwidth for a specific example
file_path = "Data/processed/HW1_intro_610300032_male_segment_1.wav"
spectral_bandwidth = extract_spectral_bandwidth(file_path)
plt.figure(figsize=(10, 5))
librosa.display.specshow(spectral_bandwidth, x_axis='time', sr=22050, cmap='coolwarm')
plt.colorbar(label="Spectral Bandwidth")
plt.title("Spectral Bandwidth")
plt.xlabel("Time")
plt.ylabel("Frequency Bands")
plt.show()
