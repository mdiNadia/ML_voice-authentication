from sklearn.manifold import TSNE
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

# کاهش ابعاد به 2 بعد با t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# نمایش در فضای دو بعدی
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.astype('category').cat.codes, cmap='viridis', alpha=0.7)
plt.title('t-SNE Projection (2D)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label="Gender")
plt.show()
