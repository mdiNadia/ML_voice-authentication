import umap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

# بارگیری داده‌های ویژگی‌های صوتی
features_df = pd.read_csv("Data/features/extract_feature.csv")

# انتخاب ویژگی‌ها و حذف ستون‌های متنی
X = features_df.drop(columns=['filename', 'student_id', 'gender'])
y = features_df['gender']  # استفاده از جنسیت برای خوشه‌بندی

# نرمال‌سازی داده‌ها
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# کاهش ابعاد به ۳ بعد با UMAP
umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
X_umap_3d = umap_3d.fit_transform(X_scaled)

# تبدیل به DataFrame برای ذخیره‌سازی
umap_df_3d = pd.DataFrame(X_umap_3d, columns=['UMAP Component 1', 'UMAP Component 2', 'UMAP Component 3'])
umap_df_3d['gender'] = y  # اضافه کردن جنسیت به DataFrame برای مرجع

# ذخیره در فایل CSV
umap_df_3d.to_csv("Data/features/classification/UMAP_reduced_features_3D.csv", index=False)

# نمایش در فضای سه‌بعدی
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], c=y.astype('category').cat.codes, cmap='viridis', alpha=0.7)

ax.set_title('UMAP Projection (3D)')
ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')
ax.set_zlabel('UMAP Component 3')

plt.colorbar(scatter, label="Gender")
plt.show()
