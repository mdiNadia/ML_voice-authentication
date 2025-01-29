import umap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# بارگیری داده‌های ویژگی‌های صوتی
features_df = pd.read_csv("Data/features/extract_feature.csv")

# انتخاب ویژگی‌ها و حذف ستون‌های متنی
X = features_df.drop(columns=['filename', 'student_id', 'gender'])
y = features_df['gender']  # استفاده از جنسیت برای خوشه‌بندی

# نرمال‌سازی داده‌ها
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# کاهش ابعاد به ۲ بعد با UMAP
umap_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
X_umap_2d = umap_2d.fit_transform(X_scaled)

# تبدیل به DataFrame برای ذخیره‌سازی
umap_df = pd.DataFrame(X_umap_2d, columns=['UMAP Component 1', 'UMAP Component 2'])
umap_df['gender'] = y  # اضافه کردن جنسیت به DataFrame برای مرجع

# ذخیره در فایل CSV
umap_df.to_csv("Data/features/classification/UMAP_reduced_features.csv", index=False)

# نمایش در فضای دو بعدی
plt.figure(figsize=(8, 6))
plt.scatter(X_umap_2d[:, 0], X_umap_2d[:, 1], c=y.astype('category').cat.codes, cmap='plasma', alpha=0.7)
plt.colorbar(label="Gender")
plt.title('UMAP Projection (2D)')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()
