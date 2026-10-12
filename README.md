# Smart Waste Classification System

A Python college mini-project that classifies a single waste item from a webcam photo or an uploaded image. It uses transfer learning with MobileNetV2 and displays prediction confidence, disposal guidance, and saved prediction history.

## Features

- Capture an image using the browser webcam or upload JPG/PNG.
- Train a classifier for any folder-based set of waste categories.
- Predict a category and show the top three probabilities.
- Show a warning for low-confidence predictions.
- Save prediction events in `outputs/predictions.csv`.
- View category counts and download prediction history.

> This is a student prototype. It does not physically sort waste, identify every material type, or determine local recycling eligibility with certainty. Follow local disposal rules.

## 1. Requirements

- Windows 10/11
- Python 3.11 recommended
- Internet access for installing packages and downloading MobileNetV2 pretrained weights on the first training run
- A webcam for camera capture (optional; image upload also works)

TensorFlow's official installation page lists supported Python versions and installation instructions: https://www.tensorflow.org/install

## 2. Setup on Windows PowerShell

Open this project folder in VS Code and run:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If Python 3.11 is not installed, check available versions using `py -0p` and choose a Python version supported by the installed TensorFlow release. If an application-control policy blocks package installation, do not disable the policy; use an approved Python installation or environment.

## 3. Get the dataset

A common starter dataset is TrashNet, with six labels: cardboard, glass, metal, paper, plastic, and trash. The original dataset contains 2,527 images and is available here:

- Original project/dataset: https://github.com/garythung/trashnet
- Resized dataset mirror with class folders: https://github.com/Ebuka24Stephen/trashnet-dataset

Review the dataset source's license/terms and cite it in your report. Download the dataset and copy the image files into this project's `dataset` directory so its layout is:

```text
dataset/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

Each category folder should contain its images directly. Do not leave an extra nested folder level between `dataset` and the class folders. You can use fewer categories, but each category must have a subfolder and enough representative images.

## 4. Train the model

From the project folder, run:

```powershell
.\venv\Scripts\python.exe train_model.py --epochs 8
```

Optional: give a custom dataset folder or training duration:

```powershell
.\venv\Scripts\python.exe train_model.py --data dataset --epochs 10 --batch-size 16
```

The script splits the folder-based dataset into training and validation subsets (80/20), trains a small classification head on top of frozen MobileNetV2 features, and saves:

- `models/waste_classifier.keras`
- `models/class_names.json`

Do not run the web app expecting predictions until these files have been created. Training duration varies by laptop. If memory is limited, use `--batch-size 8` or `16`.

## 5. Run the app

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser. Allow camera access if you choose **Use webcam**. Alternatively, choose **Upload image**.

## 6. Project workflow

1. User takes a webcam photo or uploads an image.
2. The app converts it to RGB and resizes it to 224 × 224 pixels.
3. MobileNetV2-based model predicts the trained categories.
4. The UI shows the most likely class, its confidence, and the top three predictions.
5. The prediction is appended to `outputs/predictions.csv`.
6. The History & analytics tab displays category counts and offers CSV export.

## 7. Troubleshooting

- **Model not found:** run `train_model.py` and verify both files exist in `models/`.
- **No class folders found:** ensure `dataset/cardboard`, `dataset/glass`, etc. contain image files directly.
- **Camera is blank or unavailable:** allow browser camera permissions, close other apps using the camera, or use image upload.
- **Training is slow / out of memory:** lower `--batch-size` to 8 or 16 and reduce epochs.
- **Incorrect predictions:** use a clearer image with one object, improve lighting, add more representative training images, and evaluate on images not used in training.
- **Confidence is low:** treat the result as uncertain; the app intentionally displays a warning under 55% confidence.

## 8. Viva summary

The Smart Waste Classification System uses Python, Streamlit, TensorFlow/Keras, and MobileNetV2 transfer learning to classify waste images. It accepts webcam snapshots or uploaded images, preprocesses them to the model's input size, predicts a waste category, and stores prediction history for simple analytics. The model is trained from labeled image folders and can be improved by adding varied training images and evaluating on a separate test set.

## Suggested report citations

- Thung, G. and Yang, M. (2016), *Classification of Trash for Recyclability Status*, Stanford CS229 project report and TrashNet dataset: https://github.com/garythung/trashnet
- TensorFlow installation guide: https://www.tensorflow.org/install
- Streamlit camera input documentation: https://docs.streamlit.io/develop/api-reference/widgets/st.camera_input
