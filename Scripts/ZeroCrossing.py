import librosa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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
        zcr = librosa.feature.zero_crossing_rate(y=y)

        return np.mean(zcr)  # بازگرداندن مقدار میانگین
    except Exception as e:
        print(f"❌ خطا در پردازش {file_path}: {e}")
        return None

# 📝 پردازش فایل‌ها و استخراج ویژگی‌ها
data = []
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

for file in audio_files:
    file_path = os.path.join(audio_folder, file)

    # استخراج Zero-Crossing Rate
    zcr_feature = extract_zero_crossing_rate(file_path)
    if zcr_feature is not None:
        # استخراج اطلاعات از نام فایل
        parts = file.split('_')
        if len(parts) >= 4:  # بررسی نام‌گذاری صحیح
            student_id = parts[2]  # شماره دانشجویی
            gender = parts[3].split('.')[0]  # جنسیت (male/female)
            
            # ذخیره اطلاعات در لیست
            data.append({"filename": file, "student_id": student_id, "gender": gender, "zcr": zcr_feature})

# 📊 تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# 📁 ذخیره در فایل CSV
df.to_csv(output_csv, index=False)
print(f"✅ ویژگی‌های Zero-Crossing Rate در '{output_csv}' ذخیره شد.")


# Plot Zero-Crossing Rate for a specific example
# 📂 انتخاب یک فایل برای رسم نمودار
file_path = "Data/processed/HW1_intro_610300032_male_segment_1.wav"

# استخراج ویژگی‌ها
y, sr = librosa.load(file_path, sr=22050)
zcr = librosa.feature.zero_crossing_rate(y=y)

# رسم نمودار صحیح برای ZCR
plt.figure(figsize=(10, 4))
plt.plot(zcr[0], label="Zero Crossing Rate", color="blue")
plt.xlabel("Frames")
plt.ylabel("Zero Crossing Rate")
plt.title("Zero-Crossing Rate over Time")
plt.legend()
plt.grid(True)
plt.show()

