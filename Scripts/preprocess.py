#حذف نویز با روش spectral_subtraction و کتابخانه  librosa
import librosa
import numpy as np
import librosa.display
import os
import glob
import matplotlib.pyplot as plt
import soundfile as sf  # برای ذخیره فایل

def spectral_subtraction(input_file, output_file, noise_duration=1.0, sr=16000):
    # 1. بارگذاری فایل صوتی
    audio, sr = librosa.load(input_file, sr=sr)

    # 2. استخراج نمونه‌های نویز (Noise Profile)
    noise_samples = int(noise_duration * sr)  # نویز اول فایل (مثلاً 1 ثانیه)
    noise_audio = audio[:noise_samples]
    
    # 3. محاسبه Short-Time Fourier Transform (STFT)
    audio_stft = librosa.stft(audio, n_fft=1024, hop_length=512)
    noise_stft = librosa.stft(noise_audio, n_fft=1024, hop_length=512)
    
    # 4. محاسبه قدرت (Magnitude) و میانگین طیف نویز
    noise_magnitude = np.mean(np.abs(noise_stft), axis=1)  # میانگین طیف نویز
    
    # 5. کم کردن طیف نویز از طیف اصلی
    audio_magnitude = np.abs(audio_stft)
    clean_magnitude = np.maximum(audio_magnitude - noise_magnitude[:, np.newaxis], 0)  # جلوگیری از مقادیر منفی
    
    # حفظ فاز اصلی
    phase = np.angle(audio_stft)
    clean_stft = clean_magnitude * np.exp(1j * phase)
    
    # 6. تبدیل دوباره به دامنه زمانی
    clean_audio = librosa.istft(clean_stft, hop_length=512)
    
    # 7. ذخیره فایل خروجی با استفاده از کتابخانه soundfile
    sf.write(output_file, clean_audio, sr)
    
    # 8. نمایش سیگنال اصلی و تمیز شده
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr, alpha=0.5)
    plt.title("Original Audio")
    
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(clean_audio, sr=sr, alpha=0.5, color='orange')
    plt.title("Denoised Audio")
    plt.tight_layout()
    plt.show()

def process_directory(input_directory , output_directory , noise_duration=1.0, sr=16000):
    # 1. گرفتن تمام فایل‌های صوتی در پوشه ورودی
    audio_files = glob.glob(os.path.join(input_directory, "*.mp3"))
    
    for audio_file in audio_files:
        # 2. ایجاد نام فایل خروجی
        output_file = os.path.join(output_directory, os.path.basename(audio_file))
        
        # 3. اعمال حذف نویز
        spectral_subtraction(audio_file, output_file, noise_duration, sr)
        print(f"Processed: {audio_file} -> {output_file}")

# مشخص کردن پوشه ورودی و خروجی
input_directory = './Data/raw'  # مسیر پوشه فایل‌های صوتی ورودی
output_directory = './Data/processed' # مسیر پوشه برای ذخیره فایل‌های خروجی

# اطمینان از وجود پوشه‌های خروجی
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# پردازش تمام فایل‌ها
process_directory(input_directory, output_directory)

#End of denoise_audio
def resample_audio(audio, original_sr, target_sr=16000):
    return ""

def normalize_audio(audio):
    return ""

def Windowing_audio(input_file, target_sr=16000):
    return ""

