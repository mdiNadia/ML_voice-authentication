from sklearn.preprocessing import LabelEncoder
from classification_data import X_train_gender, X_test_gender, y_train_gender, y_test_gender, X_train_identity, X_test_identity, y_train_identity, y_test_identity
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# استفاده از LabelEncoder برای تبدیل کلاس‌ها به اعداد
label_encoder_gender = LabelEncoder()
label_encoder_identity = LabelEncoder()

# تبدیل y_train و y_test برای جنسیت به اعداد (0 و 1)
y_train_gender_encoded = label_encoder_gender.fit_transform(y_train_gender)
y_test_gender_encoded = label_encoder_gender.transform(y_test_gender)

# تبدیل y_train و y_test برای هویت به اعداد
y_train_identity_encoded = label_encoder_identity.fit_transform(y_train_identity)
y_test_identity_encoded = label_encoder_identity.transform(y_test_identity)

# لیست مدل‌ها برای آزمایش
models = {
    "SVM": SVC(probability=True),  # برای محاسبه AUC
    "KNN": KNeighborsClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "MLP": MLPClassifier(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
}

# آموزش و ارزیابی هر مدل برای جنسیت
for name, model in models.items():
    print(f"\nTraining {name} for Gender...")

    # آموزش مدل با داده‌های آموزش جنسیت (y_train_gender_encoded)
    model.fit(X_train_gender, y_train_gender_encoded)

    # پیش‌بینی با استفاده از داده‌های تست جنسیت (y_test_gender_encoded)
    y_pred_gender = model.predict(X_test_gender)
    
    # تبدیل دوباره پیش‌بینی‌ها به مقادیر اصلی جنسیت
    y_pred_gender_original = label_encoder_gender.inverse_transform(y_pred_gender)

    # محاسبه ماتریس آشفتگی برای جنسیت
    cm_gender = confusion_matrix(y_test_gender, y_pred_gender_original)
    print(f"\n{name} - Gender Confusion Matrix:")
    print(cm_gender)
    
    # رسم ماتریس آشفتگی برای جنسیت
    sns.heatmap(cm_gender, annot=True, fmt="d", cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"{name} - Gender Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # گزارش classification برای جنسیت
    print(f"\n{name} - Gender Classification Report:")
    print(classification_report(y_test_gender, y_pred_gender_original))

    # دقت کلی برای جنسیت
    accuracy_gender = accuracy_score(y_test_gender, y_pred_gender_original)
    print(f"{name} - Gender Accuracy: {accuracy_gender:.4f}")
    
    # ROC AUC برای جنسیت
    if hasattr(model, "predict_proba"):
        auc_score_gender = roc_auc_score(y_test_gender_encoded, model.predict_proba(X_test_gender)[:, 1])
        print(f"{name} - Gender ROC AUC Score: {auc_score_gender:.4f}")

    # تحلیل میزان خطای هر کلاس برای جنسیت
    print(f"{name} - Gender Error Analysis:")
    class_errors_gender = 1 - accuracy_score(y_test_gender, y_pred_gender_original)
    print(f"  Overall Gender Error: {class_errors_gender:.4f}")

    # برای تحلیل خطای هر کلاس جنسیت می‌توانیم Precision، Recall و F1-Score را از classification report استخراج کنیم.
    report_gender = classification_report(y_test_gender, y_pred_gender_original, output_dict=True)
    for label in report_gender.keys():
        if label != 'accuracy':
            print(f"Class {label}:")
            print(f"Precision: {report_gender[label]['precision']:.4f}")
            print(f"Recall: {report_gender[label]['recall']:.4f}")
            print(f"F1-Score: {report_gender[label]['f1-score']:.4f}")


# آموزش و ارزیابی هر مدل برای هویت
for name, model in models.items():
    print(f"\nTraining {name} for Identity...")

    # آموزش مدل با داده‌های آموزش هویت (y_train_identity_encoded)
    model.fit(X_train_identity, y_train_identity_encoded)

    # پیش‌بینی با استفاده از داده‌های تست هویت (y_test_identity_encoded)
    y_pred_identity = model.predict(X_test_identity)
    
    # تبدیل دوباره پیش‌بینی‌ها به مقادیر اصلی هویت
    y_pred_identity_original = label_encoder_identity.inverse_transform(y_pred_identity)

    # محاسبه ماتریس آشفتگی برای هویت
    cm_identity = confusion_matrix(y_test_identity, y_pred_identity_original)
    print(f"\n{name} - Identity Confusion Matrix:")
    print(cm_identity)
    
    # رسم ماتریس آشفتگی برای هویت
    sns.heatmap(cm_identity, annot=True, fmt="d", cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"{name} - Identity Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # گزارش classification برای هویت
    print(f"\n{name} - Identity Classification Report:")
    print(classification_report(y_test_identity, y_pred_identity_original))

    # دقت کلی برای هویت
    accuracy_identity = accuracy_score(y_test_identity, y_pred_identity_original)
    print(f"{name} - Identity Accuracy: {accuracy_identity:.4f}")
    
    # ROC AUC برای هویت
    if hasattr(model, "predict_proba"):
        # برای چندکلاس: استفاده از پارامتر multi_class به 'ovr'
        auc_score_identity = roc_auc_score(y_test_identity_encoded, model.predict_proba(X_test_identity), multi_class='ovr', average='macro')
        print(f"{name} - Identity ROC AUC Score: {auc_score_identity:.4f}")

    # تحلیل میزان خطای هر کلاس برای هویت
    print(f"{name} - Identity Error Analysis:")
    class_errors_identity = 1 - accuracy_score(y_test_identity, y_pred_identity_original)
    print(f"  Overall Identity Error: {class_errors_identity:.4f}")

    # برای تحلیل خطای هر کلاس هویت می‌توانیم Precision، Recall و F1-Score را از classification report استخراج کنیم.
    report_identity = classification_report(y_test_identity, y_pred_identity_original, output_dict=True)
    for label in report_identity.keys():
        if label != 'accuracy':
            print(f"Class {label}:")
            print(f"Precision: {report_identity[label]['precision']:.4f}")
            print(f"Recall: {report_identity[label]['recall']:.4f}")
            print(f"F1-Score: {report_identity[label]['f1-score']:.4f}")
