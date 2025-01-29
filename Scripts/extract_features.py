import pandas as pd
import glob
import matplotlib.pyplot as plt

# 1. خواندن و ادغام ویژگی‌ها از چند فایل CSV
feature_files = glob.glob("Data/features/*.csv")

# بررسی اینکه آیا فایل‌ها موجود هستند یا نه
if not feature_files:
    raise FileNotFoundError("⛔ هیچ فایلی در مسیر مشخص‌شده پیدا نشد!")

df_list = [pd.read_csv(file) for file in feature_files]

df_final = df_list[0]
for df in df_list[1:]:
    df_final = pd.merge(df_final, df, on=['filename', 'student_id', 'gender'], how='left', suffixes=('', '_dup'))

# حذف ستون‌های تکراری که پسوند `_dup` دارند
df_final = df_final.loc[:, ~df_final.columns.str.endswith('_dup')]

df_final.to_csv("Data/features/extract_feature.csv", index=False)
print("✅ تمامی ویژگی‌ها با موفقیت ترکیب و ذخیره شدند!")

