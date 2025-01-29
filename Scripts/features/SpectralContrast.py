import librosa
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

        # 🔹 3. محاسبه Spectral Contrast
        stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
        spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=n_bands)

        return np.mean(spectral_contrast, axis=1)  # میانگین هر باند فرکانسی
    except Exception as e:
        print(f"❌ خطا در پردازش {file_path}: {e}")
        return None

# 📝 پردازش فایل‌ها و استخراج ویژگی‌ها
data = []
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)

    # استخراج Spectral Contrast
    spectral_contrast_features = extract_spectral_contrast(file_path)
    if spectral_contrast_features is not None:
        # استخراج اطلاعات از نام فایل
        parts = file.split('_')
        if len(parts) >= 4:
            student_id = parts[2]  # شماره دانشجویی
            gender = parts[3].split('.')[0]  # جنسیت (male/female)

            # ذخیره اطلاعات در لیست
            feature_dict = {"filename": file, "student_id": student_id, "gender": gender}
            for i in range(len(spectral_contrast_features)):
                feature_dict[f'spectral_contrast_{i+1}'] = spectral_contrast_features[i]

            data.append(feature_dict)

# 📊 تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# 📁 ذخیره در فایل CSV
df.to_csv(output_csv, index=False)
print(f"✅ ویژگی‌های Spectral Contrast در '{output_csv}' ذخیره شد.")



# 📂 انتخاب یک فایل برای رسم نمودار
file_path = "Data/processed/HW1_intro_610300032_male_segment_1.wav"

# استخراج ویژگی‌ها
y, sr = librosa.load(file_path, sr=22050)
stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=6)

# رسم Spectral Contrast
plt.figure(figsize=(10, 5))
librosa.display.specshow(spectral_contrast, x_axis='time', sr=sr, cmap='coolwarm')
plt.colorbar(label="Spectral Contrast")
plt.title("Spectral Contrast over Time")
plt.xlabel("Time")
plt.ylabel("Frequency Bands")
plt.show()


