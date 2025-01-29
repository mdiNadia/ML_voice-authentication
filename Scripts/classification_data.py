import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# بارگذاری داده‌های کاهش‌یافته PCA برای جنسیت
pca_gender_df = pd.read_csv('Data/features/classification/pca_reduced_features_gender.csv')

# بررسی داده‌های گمشده و پر کردن مقدارهای گمشده اگر باشد
numeric_cols = pca_gender_df.select_dtypes(include=['float64', 'int64']).columns
pca_gender_df[numeric_cols] = pca_gender_df[numeric_cols].fillna(pca_gender_df[numeric_cols].mean())

# انتخاب ویژگی‌ها و برچسب‌ها برای جنسیت
X_gender_pca = pca_gender_df[['PCA_Component_1', 'PCA_Component_2']]  # انتخاب ویژگی‌های PCA
y_gender = pca_gender_df['gender']  # برچسب‌های جنسیت

# حذف کلاس‌هایی که تنها یک نمونه دارند
class_counts_gender = dict(zip(*np.unique(y_gender, return_counts=True)))
min_classes_gender = [class_ for class_, count in class_counts_gender.items() if count < 2]
X_gender_filtered = X_gender_pca[~y_gender.isin(min_classes_gender)]
y_gender_filtered = y_gender[~y_gender.isin(min_classes_gender)]

# تقسیم داده‌ها به دو مجموعه آموزش و تست (با رعایت توزیع کلاس‌ها برای جنسیت)
X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(X_gender_filtered, y_gender_filtered, test_size=0.25, stratify=y_gender_filtered, random_state=42)

# بررسی توزیع کلاس‌ها در مجموعه‌های آموزشی و تست برای جنسیت
print("Gender - Training set class distribution:", dict(zip(*np.unique(y_train_gender, return_counts=True))))
print("Gender - Test set class distribution:", dict(zip(*np.unique(y_test_gender, return_counts=True))))


# بارگذاری داده‌های کاهش‌یافته PCA برای هویت
pca_identity_df = pd.read_csv('Data/features/classification/pca_reduced_features_identity.csv')

# بررسی داده‌های گمشده و پر کردن مقدارهای گمشده اگر باشد
pca_identity_df[numeric_cols] = pca_identity_df[numeric_cols].fillna(pca_identity_df[numeric_cols].mean())

# انتخاب ویژگی‌ها و برچسب‌ها برای هویت
X_identity_pca = pca_identity_df[['PCA_Component_1', 'PCA_Component_2']]  # انتخاب ویژگی‌های PCA
y_identity = pca_identity_df['student_id']  # برچسب‌های هویت

# حذف کلاس‌هایی که تنها یک نمونه دارند
class_counts_identity = dict(zip(*np.unique(y_identity, return_counts=True)))
min_classes_identity = [class_ for class_, count in class_counts_identity.items() if count < 2]
X_identity_filtered = X_identity_pca[~y_identity.isin(min_classes_identity)]
y_identity_filtered = y_identity[~y_identity.isin(min_classes_identity)]

# تقسیم داده‌ها به دو مجموعه آموزش و تست (با رعایت توزیع کلاس‌ها برای هویت)
X_train_identity, X_test_identity, y_train_identity, y_test_identity = train_test_split(X_identity_filtered, y_identity_filtered, test_size=0.25, stratify=y_identity_filtered, random_state=42)

# بررسی توزیع کلاس‌ها در مجموعه‌های آموزشی و تست برای هویت
print("Identity - Training set class distribution:", dict(zip(*np.unique(y_train_identity, return_counts=True))))
print("Identity - Test set class distribution:", dict(zip(*np.unique(y_test_identity, return_counts=True))))
