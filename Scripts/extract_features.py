import os
import librosa
import numpy as np
import pandas as pd

# مسیر پوشه شامل فایل‌های صوتی
audio_folder = "Data/processed"

# تابع استخراج ویژگی‌های صوتی
def extract_features(file_path):
    features = {}
    try:
        # بارگذاری فایل صوتی
        y, sr = librosa.load(file_path, sr=16000)

        # ویژگی‌های فرکانسی
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

        # ضرایب MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}'] = np.mean(mfccs[i])

        # ویژگی‌های زمانی
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y=y))
        features['energy'] = np.sum(y**2) / len(y)
    
    except Exception as e:
        print(f"خطا در پردازش فایل {file_path}: {e}")

    return features

# تابع استخراج شماره دانشجویی و جنسیت از نام فایل
def parse_filename(file_path):
    file_name = os.path.basename(file_path)
    parts = file_name.split("_")
    
    student_id = None
    gender = None

    # جستجوی شماره دانشجویی (اعداد 9 رقمی) و جنسیت
    for part in parts:
        if part.isdigit() and len(part) == 9:
            student_id = part
        elif part in ["male", "female"]:
            gender = part

    return student_id, gender

# خواندن فایل‌های صوتی از پوشه
audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(".wav")]

# پردازش فایل‌ها و استخراج ویژگی‌ها
data = []
for file_path in audio_files:
    student_id, gender = parse_filename(file_path)  # استخراج اطلاعات از نام فایل
    
    if student_id and gender:  # فقط پردازش فایل‌های معتبر
        features = extract_features(file_path)
        features['filename'] = os.path.basename(file_path)
        features['student_id'] = student_id
        features['gender'] = gender
        data.append(features)

# تبدیل به DataFrame
df = pd.DataFrame(data)

# ذخیره دیتافریم به CSV
output_path = "Data/features/extracted_features.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"ویژگی‌ها استخراج و در فایل '{output_path}' ذخیره شدند.")
