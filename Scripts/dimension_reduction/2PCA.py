import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# بارگذاری داده‌ها
features_df = pd.read_csv('Data/features/extract_feature.csv')

# بررسی داده‌های گمشده و پر کردن مقدارهای گمشده
numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())

# 1. بر اساس جنسیت داده‌ها را تقسیم می‌کنیم
# انتخاب ویژگی‌ها و برچسب‌ها
X_gender = features_df.drop(columns=['filename', 'student_id', 'gender'])
y_gender = features_df['gender']

# نرمال‌سازی
scaler = MinMaxScaler()
X_gender_scaled = scaler.fit_transform(X_gender)

# کاهش ابعاد با PCA
pca_gender = PCA(n_components=2)
X_gender_pca = pca_gender.fit_transform(X_gender_scaled)

# نمایش واریانس توضیح داده‌شده
explained_variance_gender = pca_gender.explained_variance_ratio_
print("Explained Variance Ratio for Gender:", explained_variance_gender)

# ایجاد DataFrame جدید
pca_gender_df = pd.DataFrame(X_gender_pca, columns=['PCA_Component_1', 'PCA_Component_2'])
pca_gender_df['student_id'] = features_df['student_id']
pca_gender_df['gender'] = features_df['gender']

pca_gender_df.to_csv('Data/features/classification/pca_reduced_features_gender.csv', index=False)
print("✅ داده‌های PCA برای جنسیت با موفقیت ذخیره شدند!")

# رسم نمودار PCA برای جنسیت
plt.figure(figsize=(8, 6))
scatter_gender = plt.scatter(X_gender_pca[:, 0], X_gender_pca[:, 1], 
                             c=features_df['gender'].astype('category').cat.codes, 
                             cmap='coolwarm', alpha=0.7)

plt.colorbar(scatter_gender, label="Gender")
plt.title('PCA of Features (2 Components) for Gender')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# افزودن نام‌ها به نقاط
for i, txt in enumerate(features_df['student_id']):
    plt.annotate(txt, (X_gender_pca[i, 0], X_gender_pca[i, 1]), fontsize=8, alpha=0.5)

plt.show()


# 2. بر اساس هویت داده‌ها را تقسیم می‌کنیم
# انتخاب ویژگی‌ها و برچسب‌ها
X_identity = features_df.drop(columns=['filename', 'student_id', 'gender'])
y_identity = features_df['student_id']  # بر اساس هویت تقسیم‌بندی می‌کنیم

# نرمال‌سازی
scaler = MinMaxScaler()
X_identity_scaled = scaler.fit_transform(X_identity)

# کاهش ابعاد با PCA
pca_identity = PCA(n_components=2)
X_identity_pca = pca_identity.fit_transform(X_identity_scaled)

# نمایش واریانس توضیح داده‌شده
explained_variance_identity = pca_identity.explained_variance_ratio_
print("Explained Variance Ratio for Identity:", explained_variance_identity)

# ایجاد DataFrame جدید
pca_identity_df = pd.DataFrame(X_identity_pca, columns=['PCA_Component_1', 'PCA_Component_2'])
pca_identity_df['student_id'] = features_df['student_id']
pca_identity_df['gender'] = features_df['gender']

pca_identity_df.to_csv('Data/features/classification/pca_reduced_features_identity.csv', index=False)
print("✅ داده‌های PCA برای هویت با موفقیت ذخیره شدند!")

# رسم نمودار PCA برای هویت
plt.figure(figsize=(8, 6))
scatter_identity = plt.scatter(X_identity_pca[:, 0], X_identity_pca[:, 1], 
                               c=features_df['student_id'].astype('category').cat.codes, 
                               cmap='coolwarm', alpha=0.7)

plt.colorbar(scatter_identity, label="Identity")
plt.title('PCA of Features (2 Components) for Identity')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# افزودن نام‌ها به نقاط
for i, txt in enumerate(features_df['student_id']):
    plt.annotate(txt, (X_identity_pca[i, 0], X_identity_pca[i, 1]), fontsize=8, alpha=0.5)

plt.show()
