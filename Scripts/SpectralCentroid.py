import librosa
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

        # 🔹 2. نرمال‌سازی سیگنال صوتی
        y = librosa.util.normalize(y)

        # 🔹 3. محاسبه Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        # 🔹 4. میانگین‌گیری روی فریم‌ها
        return np.mean(spectral_centroid, axis=1)
    except Exception as e:
        print(f"❌ خطا در پردازش {file_path}: {e}")
        return None

# 📝 پردازش فایل‌ها و استخراج ویژگی‌ها
data = []
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)

    # استخراج Spectral Centroid
    spectral_centroid_features = extract_spectral_centroid(file_path)
    if spectral_centroid_features is not None:
        # بررسی فرمت نام فایل و استخراج اطلاعات
        parts = file.split('_')
        if len(parts) >= 4:
            student_id = parts[2]  # شماره دانشجویی
            gender = parts[3].split('.')[0]  # جنسیت (male/female)

            # ذخیره اطلاعات در لیست
            feature_dict = {"filename": file, "student_id": student_id, "gender": gender}
            for i in range(len(spectral_centroid_features)):
                feature_dict[f'spectral_centroid_{i+1}'] = spectral_centroid_features[i]

            data.append(feature_dict)

# 📊 تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# 📁 ذخیره در فایل CSV
df.to_csv(output_csv, index=False)
print(f"✅ ویژگی‌های Spectral Centroid در '{output_csv}' ذخیره شد.")


# 📂 انتخاب یک فایل نمونه برای رسم نمودار
file_path = "Data/processed/HW1_intro_610300032_male_segment_1.wav"

# بارگذاری سیگنال صوتی
y, sr = librosa.load(file_path, sr=22050)
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

# تنظیم محور زمان متناسب با تعداد فریم‌ها
frames = range(spectral_centroid.shape[1])
time = librosa.frames_to_time(frames, sr=sr)

# 🎨 رسم نمودار
plt.figure(figsize=(10, 5))
plt.plot(time, spectral_centroid[0], color='r', label='Spectral Centroid')
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectral Centroid Over Time")
plt.legend()
plt.show()

