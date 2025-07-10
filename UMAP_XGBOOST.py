import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import umap


def normalize_thermal_temperature_range(thermal_image, expected_min_temp, expected_max_temp):

    # Convert intensity (0–255) → temperature range
    temp_img = expected_min_temp + (img.astype(np.float32) / 255.0) * (expected_max_temp - expected_min_temp)

    # Normalize temperature to 0–1
    normalized = (temp_img - expected_min_temp) / (expected_max_temp - expected_min_temp)  # scales correctly now
    normalized_255 = (normalized * 255).astype(np.uint8)

    return normalized_255

class ThermalImageClassifier:
    def __init__(self, target_size=(64, 64), use_umap=True, umap_components=100):
        self.target_size = target_size
        self.use_umap = use_umap
        self.umap_components = umap_components
        self.models = {}
        self.results = {}
        self.label_encoder = LabelEncoder()
        self.global_stats = {'min': float('inf'), 'max': float('-inf')}
        self.reducer = None

    def load_dataset(self, base_dir):
        print(f"Loading data from {base_dir}...")
        filters = os.listdir(base_dir)
        self.data = {}
        stage_temp_ranges = {
            'STAGE1(27-120)': (27, 120),
            'Stage-2(27-120)': (27, 120),
            'stage 3(27-660)': (27, 660),
            'stage 4(150-1600)': (150, 1600)
        }

        for filter_name in filters:
            filter_path = os.path.join(base_dir, filter_name)
            if not os.path.isdir(filter_path):
                continue

            X, y = [], []
            stages = os.listdir(filter_path)

            for label, stage in enumerate(stages):
                stage_path = os.path.join(filter_path, stage)
                if not os.path.isdir(stage_path):
                    continue

                if stage not in stage_temp_ranges:
                    print(f"⚠️ Skipping unrecognized stage name: {stage}")
                    continue

                min_temp, max_temp = stage_temp_ranges[stage]

                for img_name in os.listdir(stage_path):
                    img_path = os.path.join(stage_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    img = normalize_thermal_temperature_range(img, min_temp, max_temp)
                    img_resized = cv2.resize(img, self.target_size)
                    flat = img_resized.flatten()
                    X.append(flat)
                    y.append(label)

                    self.global_stats['min'] = min(self.global_stats['min'], np.min(flat))
                    self.global_stats['max'] = max(self.global_stats['max'], np.max(flat))

            self.data[filter_name] = (np.array(X), np.array(y))

        print(f"✅ Loaded data for {len(self.data)} filters.")

    def train_and_evaluate_models(self):
        for filter_name, (X, y) in self.data.items():
            print(f"\n🛠️ Training model for filter: {filter_name}")

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            if self.use_umap:
                print("Applying UMAP to reduce dimensionality...")
                self.reducer = umap.UMAP(n_components=self.umap_components, random_state=42)
                X_scaled = self.reducer.fit_transform(X_scaled)
                print(f"UMAP reduced dimensions to {X_scaled.shape[1]}")

            y_encoded = self.label_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            model.fit(X_train, y_train)
            self.models[filter_name] = model

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            for train_idx, val_idx in skf.split(X_scaled, y_encoded):
                model.fit(X_scaled[train_idx], y_encoded[train_idx])
                preds = model.predict(X_scaled[val_idx])
                cv_scores.append(accuracy_score(y_encoded[val_idx], preds))
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            self.results[filter_name] = {
                'accuracy': acc,
                'confusion_matrix': cm,
                'classification_report': report,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }

    def predict_filtered_image(self, image_path, filter_name):
        if filter_name not in self.models:
            raise ValueError(f"No trained model found for filter: {filter_name}")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image at {image_path}")

        stage_temp_ranges = {
            'stage1_clahe_test': (27, 120),
            'stage2_clahe_test': (27, 120),
            'stage3_clahe_test': (27, 660),
            'stage4_clahe_test': (150, 1600)
        }
        
        stage = os.path.splitext(os.path.basename(image_path))[0]
        if stage not in stage_temp_ranges:
            raise ValueError(f"Unknown stage '{stage}' for temperature normalization.")

        min_temp, max_temp = stage_temp_ranges[stage]

        img = cv2.resize(img, self.target_size)
    
        img = img.astype(np.float32)
        normalized = (img - min_temp) / (max_temp - min_temp)
        # Convert intensity (0–255) → temperature range
        temp_img = min_temp + (img.astype(np.float32) / 255.0) * (max_temp - min_temp)

    # Normalize temperature to 0–1
        normalized = (temp_img - min_temp) / (max_temp - min_temp)  # scales correctly now
        normalized_255 = (normalized * 255).astype(np.uint8)


        flat = normalized_255.flatten().reshape(1, -1)
        flat = self.scaler.transform(flat)
        if self.use_umap and self.reducer is not None:
            flat = self.reducer.transform(flat)

            # 🧪 DEBUG: Print statistics
        print(f"🔍 Input stats for: {os.path.basename(image_path)}")
        print(f"  Mean: {flat.mean():.4f}")
        print(f"  Std Dev: {flat.std():.4f}")
        print(f"  Min: {flat.min():.4f}, Max: {flat.max():.4f}")

        # Optional: see first few values
        print(f"  First 10 values: {flat[0][:10]}")

        model = self.models[filter_name]
        prediction = model.predict(flat)[0]
        prediction_proba = model.predict_proba(flat)[0]
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]

        return predicted_class, prediction_proba
    
    def predict_multiple_images(self, image_folder_path, filter_used):
        results = []

        for img_file in os.listdir(image_folder_path):
            img_path = os.path.join(image_folder_path, img_file)
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue  # Skip non-image files

            try:
                predicted_class, prediction_proba = self.predict_filtered_image(img_path, filter_used)
                results.append((img_file, predicted_class, prediction_proba))

                # 🔍 Print with probabilities
                proba_str = ", ".join([f"Class {i}: {p:.4f}" for i, p in enumerate(prediction_proba)])
                print(f"{img_file} ➤ Predicted: {predicted_class} | Probabilities: {proba_str}")

            except Exception as e:
                print(f"❌ Failed to predict {img_file}: {e}")

        return results


    def print_detailed_results(self):
        print("\nDETAILED RESULTS")
        print("=" * 80)
        for filter_name, res in self.results.items():
            print(f"\nFilter: {filter_name}")
            print(f"Accuracy: {res['accuracy']:.4f}")
            print(f"Cross-validation: {res['cv_mean']:.4f} (+/- {res['cv_std'] * 2:.4f})")
            print("Classification Report:")
            print(res['classification_report'])

    def plot_confusion_matrix(self, filter_name):
        if filter_name not in self.results:
            print(f"No results for {filter_name}")
            return

        cm = self.results[filter_name]['confusion_matrix']
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {filter_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()


def main():
    base_dir = r"C:\\Users\\Asus\\Desktop\\DATASETS_FOR_FILTERS\\Filtered_Outputs"
    clf = ThermalImageClassifier()
    try:
        clf.load_dataset(base_dir)
        clf.train_and_evaluate_models()
        clf.print_detailed_results()
        for filter_name in clf.results:
            clf.plot_confusion_matrix(filter_name)
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your data directory path and ensure it contains the expected folder structure.")
    image_folder_path = r"C:\Users\Asus\Desktop\DATASETS_FOR_FILTERS\prediction_clahe"
    filter_used = "CLAHE"  # or whatever your trained filter na

    predictions = clf.predict_multiple_images(image_folder_path, filter_used)

    test_image_path = r"C:\Users\Asus\Desktop\DATASETS_FOR_FILTERS\results_bilateral\stage3_clahe_test.tif"
    filter_name = "CLAHE"
    
    try:
        stage, proba = clf.predict_filtered_image(test_image_path, filter_name)
        print(f"\n🤞 Prediction for '{filter_name}' filter:")
        print(f"Predicted Stage: {stage}")
        print("Class Probabilities:")
        print("predictions for clahe", predictions)
        for idx, p in enumerate(proba):
            print(f"  Class {idx}: {p:.4f}")
    except Exception as e:
        print(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
