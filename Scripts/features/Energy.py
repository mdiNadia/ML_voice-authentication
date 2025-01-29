import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 📂 مسیر فایل‌های صوتی
audio_folder = "Data/processed"
output_csv = "Data/features/energy.csv"

# 🎵 تابع استخراج Energy (RMS)
def extract_energy(file_path, sr=22050, hop_length=512):
    try:
        # 🔹 1. بارگذاری فایل صوتی
        y, sr = librosa.load(file_path, sr=sr)

        # 🔹 2. پردازش اولیه (نرمال‌سازی)
        y = librosa.util.normalize(y)

        # 🔹 3. محاسبه Root Mean Square (RMS) Energy
        rms_energy = librosa.feature.rms(y=y, hop_length=hop_length)

        # 🔹 4. تولید بردار زمانی بر اساس تعداد فریم‌ها
        times = librosa.frames_to_time(np.arange(rms_energy.shape[1]), sr=sr, hop_length=hop_length)

        return rms_energy, times
    except Exception as e:
        print(f"❌ خطا در پردازش {file_path}: {e}")
        return None, None

# 📝 پردازش فایل‌های صوتی و استخراج ویژگی‌ها
data = []
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)

    # استخراج Energy
    rms_energy, _ = extract_energy(file_path)
    if rms_energy is not None:
        # بررسی و استخراج شماره دانشجویی و جنسیت از نام فایل
        parts = file.split('_')
        if len(parts) >= 4:
            student_id = parts[2]  # شماره دانشجویی
            gender = parts[3].split('.')[0]  # جنسیت (male/female)

            # محاسبه میانگین ویژگی‌ها
            energy_features = np.mean(rms_energy, axis=1)

            # ذخیره اطلاعات در لیست
            feature_dict = {"filename": file, "student_id": student_id, "gender": gender}
            for i in range(len(energy_features)):
                feature_dict[f'energy_{i+1}'] = energy_features[i]

            data.append(feature_dict)

# 📊 تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# 📁 ذخیره در فایل CSV
df.to_csv(output_csv, index=False)
print(f"✅ ویژگی‌های Energy در '{output_csv}' ذخیره شد.")


# 📂 انتخاب یک فایل نمونه
file_path = "Data/processed/HW1_Q1_810102087_female_segment_0.wav"

# استخراج Energy و زمان
rms_energy, times = extract_energy(file_path)

# بررسی عدم وجود خطا
if rms_energy is not None and times is not None:
    plt.figure(figsize=(10, 5))
    plt.plot(times, rms_energy[0], color='red', linewidth=1.5, label="Energy")
    plt.fill_between(times, rms_energy[0], alpha=0.3, color="red")
    
    # تنظیمات نمودار
    plt.title("Energy (Root Mean Square - RMS)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No energy data available.")

