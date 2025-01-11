# import librosa
# import numpy as np
# import os

# def normalize_audio(input_path, output_path):
#     # بارگذاری فایل صوتی
#     y, sr = librosa.load(input_path, sr=None)
    
#     # نرمال‌سازی سیگنال
#     y_normalized = y / np.max(np.abs(y))
    
#     # ذخیره داده نرمال‌شده
#     librosa.output.write_wav(output_path, y_normalized, sr)
#     print(f"Normalized audio saved to {output_path}")

# # مثال برای اجرای تابع
# if __name__ == "__main__":
#     input_file = "data/raw/example.wav"
#     output_file = "data/processed/example_normalized.wav"
#     os.makedirs("data/processed", exist_ok=True)
#     normalize_audio(input_file, output_file)
import os
import librosa
import numpy as np
from tqdm import tqdm  # برای نمایش نوار پیشرفت

def normalize_audio(input_path, output_path):
    """
    نرمال‌سازی یک فایل صوتی و ذخیره فایل خروجی
    """
    try:
        # بارگذاری فایل صوتی
        y, sr = librosa.load(input_path, sr=None)

        # نرمال‌سازی سیگنال
        y_normalized = y / np.max(np.abs(y))

        # ذخیره فایل صوتی نرمال‌شده
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        librosa.output.write_wav(output_path, y_normalized, sr)
        print(f"Normalized audio saved to {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def batch_normalize(input_folder, output_folder):
    """
    پردازش دسته‌ای تمام فایل‌های صوتی موجود در یک پوشه
    """
    # پیمایش تمام فایل‌ها در پوشه ورودی
    for root, _, files in os.walk(input_folder):
        for file in tqdm(files):  # tqdm برای نمایش نوار پیشرفت
            if file.endswith(".mp3"):  # پردازش فقط فایل‌های mp3
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                normalize_audio(input_path, output_path)

if __name__ == "__main__":
    # پوشه ورودی (شامل فایل‌های صوتی خام)
    input_folder = "data/raw"
    # پوشه خروجی (برای ذخیره فایل‌های نرمال‌شده)
    output_folder = "data/processed"

    # اجرا
    batch_normalize(input_folder, output_folder)
    print("Batch normalization completed!")

