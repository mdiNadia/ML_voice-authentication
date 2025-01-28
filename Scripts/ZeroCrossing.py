import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 📂 مسیر فایل‌های صوتی
audio_folder = "Data/processed"
output_csv = "Data/features/zero_crossing_rate.csv"

# 🎵 تابع استخراج Zero-Crossing Rate
def extract_zero_crossing_rate(file_path, sr=22050):
    try:
        # 🔹 1. بارگذاری فایل صوتی
        y, sr = librosa.load(file_path, sr=sr)

        # 🔹 2. پردازش اولیه (نرمال‌سازی)
        y = librosa.util.normalize(y)

        # 🔹 3. محاسبه Zero-Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

        return zero_crossing_rate
    except Exception as e:
        print(f"❌ خطا در پردازش {file_path}: {e}")
        return None

# 📝 پردازش فایل‌ها و استخراج ویژگی‌ها
data = []
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)

    # استخراج Zero-Crossing Rate
    zero_crossing_rate = extract_zero_crossing_rate(file_path)
    if zero_crossing_rate is not None:
        # استخراج شماره دانشجویی از نام فایل
        parts = file.split('_')
        if len(parts) >= 4:  # Ensure correct filename format
            student_id = parts[2]  # شماره دانشجویی
            gender = parts[3].split('.')[0]  # جنسیت (male/female)
            
            # محاسبه میانگین ویژگی‌ها
            crossing_rate_features = np.mean(zero_crossing_rate, axis=1)  # میانگین‌گیری در طول زمان

            # ذخیره اطلاعات در لیست
            feature_dict = {"filename": file, "student_id": student_id}
            for i in range(len(crossing_rate_features)):
                feature_dict[f'crossing_rate_{i+1}'] = crossing_rate_features[i]

            data.append(feature_dict)

# 📊 تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# 📁 ذخیره در فایل CSV
df.to_csv(output_csv, index=False)
print(f"✅ ویژگی‌های Zero-Crossing Rate در '{output_csv}' ذخیره شد.")

# Plot Zero-Crossing Rate for a specific example
file_path = "Data/processed/HW1_intro_610300032_male_segment_1.wav"
zero_crossing_rate = extract_zero_crossing_rate(file_path)
plt.figure(figsize=(10, 5))
librosa.display.specshow(zero_crossing_rate, x_axis='time', sr=22050, cmap='coolwarm')
plt.colorbar(label="Zero Crossing Rate")
plt.title("Zero-Crossing Rate")
plt.xlabel("Time")
plt.ylabel("Frames")
plt.show()
