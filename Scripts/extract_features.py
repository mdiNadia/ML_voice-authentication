import pandas as pd
import glob

# لیست فایل‌های ویژگی که قبلاً ذخیره شده‌اند
feature_files = glob.glob("Data/features/*.csv")

# خواندن و ادغام
df_list = [pd.read_csv(file) for file in feature_files]

# ادغام بر اساس 'filename'
df_final = df_list[0]
for df in df_list[1:]:
    df_final = pd.merge(df_final, df, on="filename", how="left")

# ذخیره در فایل نهایی
df_final.to_csv("Data/extract_feature.csv", index=False)
print("✅ تمامی ویژگی‌ها با موفقیت ترکیب و ذخیره شدند!")
