import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Load data
DATA_DIR = "/run/media/princenagda/5A4E832F4E83034D/school projects/sign_data"
data = []
labels = []

for file_name in os.listdir(DATA_DIR):
    if not file_name.endswith(".csv"):
        continue  # Skip non-CSV files

    # Extract label from file name
    label = file_name.split(".")[0].strip()  # Use the file name as the label
    label = label.encode('utf-8').decode('utf-8')  # Ensure proper encoding for Hindi text
    file_path = os.path.join(DATA_DIR, file_name)

    # Read CSV data
    try:
        with open(file_path, "r") as f:
            for line in f:
                landmarks = list(map(float, line.strip().split(",")))  # Parse landmarks
                data.append(landmarks)
                labels.append(label)
    except ValueError as e:
        print(f"Skipping malformed line in {file_name}: {line.strip()}")
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")

data = np.array(data)
labels = np.array(labels)

# Verify data and labels
print(f"Total samples: {len(data)}")
print(f"Labels: {set(labels)}")

# Split into training and testing sets
if len(data) > 0:  # Ensure data is not empty
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save the model
    import joblib
    joblib.dump(clf, "sign_language_model.pkl")
else:
    print("No valid data to train the model. Please check your dataset.")
