import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 📂 مسیر فایل‌های صوتی
audio_folder = "Data/processed"
output_csv = "Data/features/energy.csv"

# 🎵 تابع استخراج Energy
def extract_energy(file_path, sr=22050):
    try:
        # 🔹 1. بارگذاری فایل صوتی
        y, sr = librosa.load(file_path, sr=sr)
        print(f"Loaded {file_path}, length: {len(y)} samples")  # افزودن پیام دیباگ

        # 🔹 2. پردازش اولیه (نرمال‌سازی)
        y = librosa.util.normalize(y)
        print(f"Normalized audio: {y[:5]}...")  # افزودن پیام دیباگ

        # 🔹 3. محاسبه Energy (جمع توان سیگنال)
        energy = librosa.feature.rms(y=y)  # Root mean square energy
        print(f"Computed energy, shape: {energy.shape}")  # افزودن پیام دیباگ

        return energy
    except Exception as e:
        print(f"❌ خطا در پردازش {file_path}: {e}")
        return None

# 📝 پردازش فایل‌ها و استخراج ویژگی‌ها
data = []
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)

    # استخراج Energy
    energy = extract_energy(file_path)
    if energy is not None:
        # استخراج شماره دانشجویی از نام فایل
        parts = file.split('_')
        if len(parts) >= 4:  # Ensure correct filename format
            student_id = parts[2]  # شماره دانشجویی
            gender = parts[3].split('.')[0]  # جنسیت (male/female)
            
            # محاسبه میانگین ویژگی‌ها
            energy_features = np.mean(energy, axis=1)  # میانگین‌گیری در طول زمان

            # ذخیره اطلاعات در لیست
            feature_dict = {"filename": file, "student_id": student_id}
            for i in range(len(energy_features)):
                feature_dict[f'energy_{i+1}'] = energy_features[i]

            data.append(feature_dict)


# 📊 تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# 📁 ذخیره در فایل CSV
df.to_csv(output_csv, index=False)
print(f"✅ ویژگی‌های Energy در '{output_csv}' ذخیره شد.")

# Plot Energy for a specific example
file_path = "Data/processed/HW1_Q1_810102087_female_segment_0.wav"
energy = extract_energy(file_path)
if energy is not None:
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(energy, x_axis='time', sr=22050, cmap='coolwarm')
    plt.colorbar(label="Energy")
    plt.title("Energy")
    plt.xlabel("Time")
    plt.ylabel("Frames")
    plt.show()
else:
    print("No energy data available.")
