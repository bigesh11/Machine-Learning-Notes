# Machine Learning Notes

Notes for the Data Science course (Machine Learning and Computer Vision): the main concepts, the algorithms, a worked example for each, and the practical code that goes with it.

## How this repository is organised
Every topic has its own folder:

```
NN_Topic_Name/
├── study_guide.md           # concept summary: what it is, how it works, formulas, pros/cons
├── example.py               # small runnable example (scikit-learn, built-in data)
└── my_handwritten_notes/    # photos/scans of my own handwritten notes for the topic
```

## Contents
| # | Topic | Section | Code |
|---|---|---|---|
| 00 | [Machine Learning Overview](00_Overview/study_guide.md) | Introduction | - |
| 01 | [Data Preprocessing](01_Data_Preprocessing/study_guide.md) | Data preprocessing | [example.py](01_Data_Preprocessing/example.py) |
| 02 | [Simple Linear Regression](02_Simple_Linear_Regression/study_guide.md) | Supervised learning: regression | [example.py](02_Simple_Linear_Regression/example.py) |
| 03 | [Multiple Linear Regression](03_Multiple_Linear_Regression/study_guide.md) | Supervised learning: regression | [example.py](03_Multiple_Linear_Regression/example.py) |
| 04 | [Polynomial Regression](04_Polynomial_Regression/study_guide.md) | Supervised learning: regression | [example.py](04_Polynomial_Regression/example.py) |
| 05 | [Support Vector Regression (SVR)](05_Support_Vector_Regression/study_guide.md) | Supervised learning: regression | [example.py](05_Support_Vector_Regression/example.py) |
| 06 | [Decision Tree](06_Decision_Tree/study_guide.md) | Supervised learning: regression and classification | [example.py](06_Decision_Tree/example.py) |
| 07 | [Random Forest](07_Random_Forest/study_guide.md) | Supervised learning: regression and classification | [example.py](07_Random_Forest/example.py) |
| 08 | [Logistic Regression](08_Logistic_Regression/study_guide.md) | Supervised learning: classification | [example.py](08_Logistic_Regression/example.py) |
| 09 | [K-Nearest Neighbors (KNN)](09_KNN/study_guide.md) | Supervised learning: classification | [example.py](09_KNN/example.py) |
| 10 | [Naive Bayes](10_Naive_Bayes/study_guide.md) | Supervised learning: classification | [example.py](10_Naive_Bayes/example.py) |
| 11 | [Support Vector Machine (SVM) and Kernel SVM](11_SVM_and_Kernel_SVM/study_guide.md) | Supervised learning: classification | [example.py](11_SVM_and_Kernel_SVM/example.py) |
| 12 | [Evaluating Supervised Models](12_Evaluating_Supervised_Models/study_guide.md) | Evaluation | [example.py](12_Evaluating_Supervised_Models/example.py) |
| 13 | [K-Means Clustering](13_KMeans_Clustering/study_guide.md) | Unsupervised learning | [example.py](13_KMeans_Clustering/example.py) |
| 14 | [Hierarchical (Agglomerative) Clustering](14_Hierarchical_Clustering/study_guide.md) | Unsupervised learning | [example.py](14_Hierarchical_Clustering/example.py) |
| 15 | [Apriori and Association Rule Mining](15_Apriori_Association_Rules/study_guide.md) | Unsupervised learning | [example.py](15_Apriori_Association_Rules/example.py) |
| 16 | [Principal Component Analysis (PCA) and Kernel PCA](16_PCA_and_Kernel_PCA/study_guide.md) | Unsupervised learning | [example.py](16_PCA_and_Kernel_PCA/example.py) |
| 17 | [Evaluating Unsupervised Models](17_Evaluating_Unsupervised_Models/study_guide.md) | Evaluation | [example.py](17_Evaluating_Unsupervised_Models/example.py) |
| 18 | [YOLO Object Detection and the License Plate Project](18_YOLO_Object_Detection/study_guide.md) | Computer vision | [example.py](18_YOLO_Object_Detection/example.py) |

Also: [CHEATSHEET.md](CHEATSHEET.md), a one-page summary of formulas and when to use each algorithm.

## Practical work
- **Titanic survival prediction** (end-to-end project with a Streamlit app): https://github.com/bigesh11/titanic-survival-predictor
- **License plate monitor** (YOLOv8 + OCR + Streamlit): https://github.com/bigesh11/license-plate-monitor

## How to run the examples
```bash
python -m venv venv
venv\Scripts\activate           # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python 02_Simple_Linear_Regression/example.py
```
Every example uses data built into scikit-learn or generated in the script, so no downloads are needed.

## Note on how these were made
The study guides and example scripts were prepared with the help of an AI assistant (Claude) as a revision aid. The handwritten notes in each `my_handwritten_notes/` folder are my own.
