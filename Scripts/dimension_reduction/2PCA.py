import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 2. بارگذاری داده‌ها
features_df = pd.read_csv('Data/features/extract_feature.csv')

# بررسی داده‌های گمشده و پر کردن مقدارهای گمشده
numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())

# انتخاب ویژگی‌ها و برچسب‌ها
X = features_df.drop(columns=['filename', 'student_id', 'gender'])
y = features_df['gender']

# نرمال‌سازی
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. کاهش ابعاد با PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# نمایش واریانس توضیح داده‌شده
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

# ایجاد DataFrame جدید
pca_df = pd.DataFrame(X_pca, columns=['PCA_Component_1', 'PCA_Component_2'])
pca_df['student_id'] = features_df['student_id']
pca_df['gender'] = features_df['gender']

#pca_df.to_csv('Data/features/extract_feature_pca.csv', index=False)
#print("✅ داده‌های PCA با موفقیت ذخیره شدند!")

# رسم نمودار PCA
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                      c=features_df['gender'].astype('category').cat.codes, 
                      cmap='coolwarm', alpha=0.7)

plt.colorbar(scatter, label="Gender")
plt.title('PCA of Features (2 Components)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# افزودن نام‌ها به نقاط
for i, txt in enumerate(features_df['student_id']):
    plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.5)

plt.show()

