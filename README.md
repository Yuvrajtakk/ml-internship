# AI/ML Internship Learning Journal

**Yuvraj Tak** | B.Tech CSE-AI (2023-2027) | Anand International College of Engineering, Jaipur  
**Remote AI/ML Internship** at Watsoo Express Pvt. Ltd.  
**Mentor:** Ankit Gupta

This repository is the main record of my two-month AI/ML internship learning journey. It contains explanations, experiments, source code, notebooks, model visualizations, and practical computer-vision work.

> The purpose of this repository is not only to collect working code. It is to make the reasoning behind each model visible: what problem it solves, what assumptions it makes, how it learns, how it is evaluated, and where it can fail.

## Table of contents

- [Project overview](#project-overview)
- [Learning roadmap](#learning-roadmap)
- [Track 1: ML algorithms](#track-1-ml-algorithms)
- [Track 2: YOLO computer vision](#track-2-yolo-computer-vision)
- [Complete visual gallery](#complete-visual-gallery)
- [Repository map](#repository-map)
- [How to run](#how-to-run)
- [What I learned](#what-i-learned)
- [Resources](#resources)

## Project overview

The repository has two connected tracks:

1. **Classical machine learning:** learn the core mechanics of prediction, error, optimization, regression, classification, ensembles, and evaluation using Python, NumPy, Matplotlib, and scikit-learn.
2. **YOLO computer vision:** study the YOLO family through practical notebooks, object-detection experiments, vehicle tracking, and a complete local YOLOv6 source tree.

The repeated application theme is Watsoo Express truck data. Truck distance, fuel cost, speed, vibration, engine temperature, and oil pressure make abstract ML ideas easier to connect to a real operational problem such as fleet monitoring and vehicle health prediction.

## Learning roadmap

The learning path moves from the smallest possible learning example to complete vision pipelines:

```text
Understand one weight and one error
              |
              v
Learn regression and binary classification
              |
              v
Compare trees, forests, KNN, SVM, and Naive Bayes
              |
              v
Evaluate models with precision, recall, F1, and confusion matrices
              |
              v
Study image detection with the YOLO family
              |
              v
Train, evaluate, infer, deploy, and quantize YOLOv6
```

## Track 1: ML algorithms

Location: [`ML algo/`](ML%20algo/)

The [`CODE BOOK`](ML%20algo/CODE%20BOOK/) contains one focused lesson per algorithm. The code is written as a progression: first understand the mechanism, then use established implementations, then inspect the result visually.

### 1. Foundation: the learning loop

Code: [`00 foundation.py`](ML%20algo/CODE%20BOOK/00%20foundation.py)  
Output: [`Fundamenrtal.png`](ML%20algo/CODE%20BOOK/Fundamenrtal.png)

This lesson answers: what is a machine actually doing when it learns?

- Starts with one input, one target, and one weight.
- Calculates a prediction using `input * weight`.
- Measures the error as `target - prediction`.
- Updates the weight in the direction that reduces the error.
- Demonstrates how the learning rate controls the update size.
- Repeats the process over multiple points using stochastic gradient descent.
- Adds a bias/intercept so the learned line does not have to pass through the origin.
- Tracks mean squared error and plots the learned line against the data.

The central lesson is that the same loop appears inside more advanced training procedures: predict, measure error, update parameters, and repeat.

### 2. Linear regression: predicting a continuous value

Code: [`01_linear_regression.py`](ML%20algo/CODE%20BOOK/01_linear_regression.py)  
Output: [`linear_regresion.png`](ML%20algo/CODE%20BOOK/linear_regresion.png)

This example uses 80 synthetic Watsoo Express trucks:

- Feature: distance driven in kilometres.
- Target: fuel cost.
- Intended relationship: approximately `fuel cost = 4.5 * distance + 250`.
- Noise represents traffic, weather, driver style, and sensor variation.
- Uses an 80/20 train-test split.
- Reports RMSE and R2.
- Predicts the expected fuel cost for a new truck.
- Plots the observations and the fitted regression line.

The main takeaway is that linear regression is fast and interpretable, but assumes the relationship can be represented well by a straight line.

### 3. Logistic regression: probabilities and binary decisions

Code: [`02_logistic_regression.py`](ML%20algo/CODE%20BOOK/02_logistic_regression.py)  
Output: [`Logistic_Regresion.png`](ML%20algo/CODE%20BOOK/Logistic_Regresion.png)

This lesson changes the question from "how much?" to "which class?":

- Features: truck speed and vibration level.
- Class 0: healthy truck.
- Class 1: failing truck.
- Uses a sigmoid function to convert a linear score into a probability.
- Standardizes features before fitting the model.
- Measures accuracy, precision, recall, and F1 score.
- Compares the default threshold of `0.5` with a more sensitive threshold of `0.3`.
- Visualizes the sigmoid curve, decision boundary, and threshold trade-off.

A lower threshold can catch more failing trucks, but normally creates more false alarms. This is an important operational decision, not just a mathematical setting.

### 4. Decision trees: learning human-readable rules

Code: [`03_decision_tree.py`](ML%20algo/CODE%20BOOK/03_decision_tree.py)  
Output: [`Decision Tree.png`](ML%20algo/CODE%20BOOK/Decision%20Tree.png)

The tree classifies trucks by asking a sequence of yes/no questions:

- Features: engine temperature, vibration, and oil pressure.
- Uses Gini impurity to measure how mixed a group is.
- Selects splits that make child groups more pure.
- Limits the tree depth to control overfitting.
- Compares a controlled tree with a fully grown tree.
- Displays feature importance and human-readable decision rules.
- Shows decision regions and the tree flowchart.

Trees are easy to explain and do not require feature scaling, but deep trees can memorize their training examples.

### 5. Random forests: reducing variance with ensembles

Code: [`04_random_forest.py`](ML%20algo/CODE%20BOOK/04_random_forest.py)  
Output: [`Rabdon Forest.png`](ML%20algo/CODE%20BOOK/Rabdon%20Forest.png)

This lesson builds a committee of decision trees:

- Creates healthy, failing, and ambiguous truck examples.
- Trains a single decision tree as a baseline.
- Trains a 100-tree random forest.
- Uses bootstrap samples so trees see different rows.
- Uses feature randomness so trees consider different split candidates.
- Compares train and test performance.
- Uses out-of-bag accuracy as an internal validation estimate.
- Visualizes feature importance and ensemble decision regions.

The key idea is that different trees make different mistakes. Combining their votes generally produces a more stable model than relying on one deep tree.

### 6. K-nearest neighbours: distance-based classification

Code: [`05_knn.py`](ML%20algo/CODE%20BOOK/05_knn.py)  
Output: [`KNN implementation.png`](ML%20algo/CODE%20BOOK/KNN%20implementation.png)

KNN stores the training data and classifies a new example by the labels of its nearest neighbours:

- Uses speed and vibration as truck features.
- Demonstrates why scaling is essential when features have different units.
- Compares raw and standardized feature spaces.
- Visualizes K values of 1, 5, and 11.
- Shows the U-shaped relationship between K and validation error.
- Uses cross-validation to choose a useful K.
- Explains why small K can overfit and large K can underfit.

Unlike many models, KNN does almost no work during training. Its computation happens when making predictions.

### 7. Support vector machines: margins and kernels

Code: [`06_svm.py`](ML%20algo/CODE%20BOOK/06_svm.py)  
Output: [`SVM.png`](ML%20algo/CODE%20BOOK/SVM.png)

This lesson compares three SVM kernels on datasets designed for their strengths:

- Linear kernel for straight-line separation.
- RBF kernel for curved or blob-shaped boundaries.
- Polynomial kernel for curved, moon-shaped separation.
- Standardizes the features before fitting.
- Reports accuracy and the number of support vectors.
- Visualizes the margin, support vectors, and decision regions.
- Explains how the `C` parameter controls the trade-off between a wide margin and training errors.

SVMs focus on the examples closest to the decision boundary. Those critical examples are the support vectors.

### 8. Naive Bayes: probability under an independence assumption

Code: [`07_naive_bayes.py`](ML%20algo/CODE%20BOOK/07_naive_bayes.py)  
Output: [`Naive Bayes.png`](ML%20algo/CODE%20BOOK/Naive%20Bayes.png)

The lesson compares the three common scikit-learn variants:

- **GaussianNB:** continuous measurements such as truck sensors.
- **MultinomialNB:** non-negative integer counts such as word or event frequencies.
- **BernoulliNB:** binary features such as pass/fail checks.

It explains Bayes' theorem, the conditional-independence assumption, class probabilities, decision regions, and the situations where each variant is appropriate. Naive Bayes is fast and useful on small datasets, even though its independence assumption is often a simplification.

### 9. Evaluation metrics: looking beyond accuracy

Code: [`08_evaluation_metrics.py`](ML%20algo/CODE%20BOOK/08_evaluation_metrics.py)  
Output: [`EVALUATION METRICS.png`](ML%20algo/CODE%20BOOK/EVALUATION%20METRICS.png)

This is the final ML lesson and brings the earlier ideas together:

- Demonstrates the accuracy trap on imbalanced data.
- Shows how a model that predicts every truck as healthy can still obtain high accuracy.
- Walks through TP, TN, FP, and FN manually.
- Calculates accuracy, precision, recall, and F1 by hand and with scikit-learn.
- Trains logistic regression on 120 truck examples.
- Compares a default threshold of `0.5` with a more cautious threshold of `0.3`.
- Prints a complete classification report.
- Visualizes confusion matrices and metric trade-offs.

For truck safety, recall can matter more than accuracy because a missed failure may be much more expensive than a false alarm.

## Track 2: YOLO computer vision

Location: [`YOLO family/`](YOLO%20family/)

The YOLO track applies the same learning mindset to images. Instead of predicting a single number or class for a row of tabular data, an object detector must identify multiple objects, classify them, and estimate their bounding boxes in one image.

### YOLO experiments and notebooks

The notebook collection is in [`YOLO  1 - 26 models/`](YOLO%20family/YOLO%20%201%20-%2026%20models/). It includes practical work for:

- [`yolov1.ipynb`](YOLO%20family/YOLO%20%201%20-%2026%20models/yolov1.ipynb)
- [`yolov2_practical.ipynb`](YOLO%20family/YOLO%20%201%20-%2026%20models/yolov2_practical.ipynb)
- [`yolov3_practical.ipynb`](YOLO%20family/YOLO%20%201%20-%2026%20models/yolov3_practical.ipynb)
- [`yolov4_practical.ipynb`](YOLO%20family/YOLO%20%201%20-%2026%20models/yolov4_practical.ipynb)
- [`yolov5_practical.ipynb`](YOLO%20family/YOLO%20%201%20-%2026%20models/yolov5_practical.ipynb)
- [`yolov6_practical.ipynb`](YOLO%20family/YOLO%20%201%20-%2026%20models/yolov6_practical.ipynb)
- [`yolov7_practical.ipynb`](YOLO%20family/YOLO%20%201%20-%2026%20models/yolov7_practical.ipynb)
- [`yolov9_practical.ipynb`](YOLO%20family/YOLO%20%201%20-%2026%20models/yolov9_practical.ipynb)
- [`yolov10_practical.ipynb`](YOLO%20family/YOLO%20%201%20-%2026%20models/yolov10_practical.ipynb)
- [`yolo 11_practical.ipynb`](YOLO%20family/YOLO%20%201%20-%2026%20models/yolo%2011_practical.ipynb)
- [`vehicle_tracking_roboflow_yolov8.ipynb`](YOLO%20family/YOLO%20%201%20-%2026%20models/vehicle_tracking_roboflow_yolov8.ipynb)

The [`V1`](YOLO%20family/V1/) and [`V2`](YOLO%20family/V2/) folders contain earlier practical experiments and sample detection inputs/outputs.

### What the YOLO work covers

Across the notebooks and source tree, the work studies the main parts of an object-detection workflow:

1. Load an image or video source.
2. Resize and preprocess the input for the model.
3. Run a one-stage detector in a single forward pass.
4. Decode predicted boxes, classes, and confidence scores.
5. Apply non-maximum suppression to remove duplicate detections.
6. Compare predictions with labels during evaluation.
7. Visualize bounding boxes and confidence values.
8. Track vehicles across frames in a practical workflow.
9. Export models for different inference runtimes.

### YOLOv6 implementation

The local YOLOv6 codebase is in [`YOLOv6/`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/). It is more than a notebook example and includes the supporting pieces of a production-style detection project:

- [`tools/train.py`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/tools/train.py): training entry point.
- [`tools/eval.py`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/tools/eval.py): evaluation entry point.
- [`tools/infer.py`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/tools/infer.py): image, directory, video, and webcam inference.
- [`yolov6/models/`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/yolov6/models/): model, backbone, neck, head, loss, and end-to-end components.
- [`yolov6/data/`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/yolov6/data/): data loading, augmentation, dataset handling, and VOC conversion.
- [`yolov6/assigners/`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/yolov6/assigners/): anchor and task-aligned label assignment.
- [`deploy/`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/deploy/): ONNX, OpenVINO, TensorRT, and NCNN deployment workflows.
- [`tools/partial_quantization/`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/tools/partial_quantization/): sensitivity analysis and selective quantization.
- [`tools/qat/`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/tools/qat/): quantization-aware training and export utilities.
- [`configs/`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/configs/): model, fine-tuning, lite, RepOpt, QARepVGG, and experiment configurations.

### YOLOv6 model concepts studied

The YOLOv6 source and documentation provide examples of:

- EfficientRep backbones and Rep-PAN necks.
- Efficient decoupled detection heads.
- P5 and P6 input-resolution configurations.
- Mosaic, MixUp, HSV, affine, and letterbox augmentation.
- ATSS and task-aligned assignment strategies.
- IoU, distribution focal, classification, and distillation losses.
- Validation with COCO metrics and precision-recall metrics.
- TensorRT, ONNX Runtime, OpenVINO, NCNN, and OpenCV deployment.
- Post-training quantization, partial quantization, RepOpt, and QAT.

## Complete visual gallery

The figures are part of the learning record. They show not only final predictions, but also decision boundaries, training behavior, evaluation mistakes, and deployment-oriented YOLO results.

### Main ML outputs

| Foundation | Linear regression | Logistic regression |
| --- | --- | --- |
| ![Foundation learning loop](ML%20algo/CODE%20BOOK/Fundamenrtal.png) | ![Linear regression output](ML%20algo/CODE%20BOOK/linear_regresion.png) | ![Logistic regression output](ML%20algo/CODE%20BOOK/Logistic_Regresion.png) |

| Decision tree | Random forest | KNN |
| --- | --- | --- |
| ![Decision tree output](ML%20algo/CODE%20BOOK/Decision%20Tree.png) | ![Random forest output](ML%20algo/CODE%20BOOK/Rabdon%20Forest.png) | ![KNN output](ML%20algo/CODE%20BOOK/KNN%20implementation.png) |

| SVM | Naive Bayes | Evaluation metrics |
| --- | --- | --- |
| ![SVM output](ML%20algo/CODE%20BOOK/SVM.png) | ![Naive Bayes output](ML%20algo/CODE%20BOOK/Naive%20Bayes.png) | ![Evaluation metrics output](ML%20algo/CODE%20BOOK/EVALUATION%20METRICS.png) |

### ML rough-work outputs

The [`ROUGH CODE`](ML%20algo/ROUGH%20CODE/) folder preserves exploratory visualizations made while developing the lessons:

| Experiment | Image |
| --- | --- |
| Decision-tree exploration | ![Decision tree rough output](ML%20algo/ROUGH%20CODE/descion%20tree.png) |
| Decision boundaries and logical rules | ![Decision boundary rough output](ML%20algo/ROUGH%20CODE/Desicion%20tree_carved%20space%20boundaries%20abd%20logical%20descion%20rules%20.png) |
| Evaluation matrix | ![Evaluation matrix rough output](ML%20algo/ROUGH%20CODE/Evaluation%20Matrix.png) |
| Gaussian Naive Bayes | ![Gaussian Naive Bayes rough output](ML%20algo/ROUGH%20CODE/Gausian%20naive%20bayes.png) |
| KNN boundaries | ![KNN rough output](ML%20algo/ROUGH%20CODE/KNN.png) |
| Random forest | ![Random forest rough output](ML%20algo/ROUGH%20CODE/Random%20fores.png) |
| Random forest with 100 trees | ![Random forest 100 trees](ML%20algo/ROUGH%20CODE/random%20forest%20%28100%20trees%29.png) |
| Random forest OOB boundary | ![Random forest OOB boundary](ML%20algo/ROUGH%20CODE/Random%20forest%20Decision%20boundary%20OOB%20accuracy.png) |
| Random forest single-tree comparison | ![Random forest single-tree comparison](ML%20algo/ROUGH%20CODE/Random%20forest%20Decision%20boundary%20OOB%20accuracy%20when%20%20n_estimators=1.png) |
| Support vector machine | ![SVM rough output](ML%20algo/ROUGH%20CODE/Support%20Vecrtor%20Machine.png) |

### YOLO detection outputs

| YOLO V1 input | YOLO V1 output | YOLO V2 input |
| --- | --- | --- |
| ![YOLO V1 input](YOLO%20family/V1/bus.jpg) | ![YOLO V1 detections](YOLO%20family/V1/output.png) | ![YOLO V2 input](YOLO%20family/V2/bus.jpg) |

| YOLO V1 resized input | YOLO V2 resized input | YOLO family sample |
| --- | --- | --- |
| ![YOLO V1 resized input](YOLO%20family/V1/bus_448.jpg) | ![YOLO V2 resized input](YOLO%20family/V2/bus_448.jpg) | ![YOLO family sample](YOLO%20family/YOLO%20%201%20-%2026%20models/bus.jpg) |

### YOLOv6 assets and model visuals

| Detection example | Training batch | VOC loss curve |
| --- | --- | --- |
| ![YOLOv6 detection example](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/assets/image3.jpg) | ![YOLOv6 training batch](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/assets/train_batch.jpg) | ![YOLOv6 VOC loss curve](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/assets/voc_loss_curve.jpg) |

| YOLOv6 overview | YOLOv6 speed comparison | YOLOv6 speed comparison v2 |
| --- | --- | --- |
| ![YOLOv6 overview](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/assets/banner-YOLO.png) | ![YOLOv6 speed comparison](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/assets/speed_comparision_v3.png) | ![YOLOv6 speed comparison version 2](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/assets/speed_comparision_v2.png) |

| YOLOv5 comparison | YOLOv6 comparison | YOLOX comparison |
| --- | --- | --- |
| ![YOLOv5 comparison](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/assets/yolov5s.jpg) | ![YOLOv6 comparison](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/assets/yolov6s.jpg) | ![YOLOX comparison](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/assets/yoloxs.jpg) |

| YOLOv6 Lite NCNN | YOLOv6 additional visual |
| --- | --- |
| ![YOLOv6 Lite NCNN](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/assets/yolov6lite_l_ncnn.jpg) | ![YOLOv6 additional visual](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/assets/picture.png) |

The YOLOv6 repository also contains sample inference images in [`data/images`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/data/images/) and deployment examples in [`deploy/ONNX/OpenCV`](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/deploy/ONNX/OpenCV/).

## Repository map

```text
ML algo/
  CODE BOOK/
    00 foundation.py              Core ML learning loop from scratch
    01_linear_regression.py       Continuous truck fuel-cost prediction
    02_logistic_regression.py     Truck health classification
    03_decision_tree.py           Explainable rule-based classification
    04_random_forest.py           Ensemble classification and OOB validation
    05_knn.py                     Distance-based classification and scaling
    06_svm.py                     Margins and three kernel variants
    07_naive_bayes.py             Gaussian, Multinomial, and Bernoulli NB
    08_evaluation_metrics.py      Accuracy, precision, recall, and F1
    *.png                         Main lesson outputs
  ROUGH CODE/
    01_basics.py                  Early experiments
    *.png                         Exploratory figures
YOLO family/
  V1/                             Early YOLO experiments and outputs
  V2/                             YOLO experiments and outputs
  YOLO  1 - 26 models/
    *.ipynb                        Practical notebooks across YOLO versions
    YOLOv6/
      assets/                      Detection, training, and benchmark images
      configs/                     Model and experiment configurations
      data/                        Dataset configuration and sample images
      deploy/                      ONNX, OpenVINO, TensorRT, and NCNN tools
      docs/                        Training, evaluation, and deployment guides
      tools/                       Train, eval, infer, and quantization tools
      yolov6/                      Core model, data, loss, and utility code
```

## How to run

### ML lessons

From the repository root, install the lightweight ML dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy matplotlib scikit-learn
```

Run individual lessons:

```bash
python "ML algo/CODE BOOK/00 foundation.py"
python "ML algo/CODE BOOK/01_linear_regression.py"
python "ML algo/CODE BOOK/02_logistic_regression.py"
python "ML algo/CODE BOOK/03_decision_tree.py"
python "ML algo/CODE BOOK/04_random_forest.py"
python "ML algo/CODE BOOK/05_knn.py"
python "ML algo/CODE BOOK/06_svm.py"
python "ML algo/CODE BOOK/07_naive_bayes.py"
python "ML algo/CODE BOOK/08_evaluation_metrics.py"
```

The scripts print metrics and open Matplotlib visualizations. The examples use synthetic data generated in the scripts, so no external dataset download is needed for the ML track.

### YOLO notebooks

Open the notebooks in VS Code or Jupyter. The notebooks may require additional framework-specific packages and model weights. Check each notebook's first cells for its exact environment and input requirements.

### YOLOv6

YOLOv6 has its own dependency file and workflow. Start with the [YOLOv6 README](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/README.md), then use the [requirements file](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/requirements.txt) in a compatible Python environment.

Typical YOLOv6 commands are run from the YOLOv6 directory:

```bash
cd "YOLO family/YOLO  1 - 26 models/YOLOv6"
pip install -r requirements.txt
python tools/infer.py --weights yolov6s.pt --source data/images
```

Training and evaluation require an appropriately prepared dataset, model weights, and usually a CUDA-capable environment. See the [custom-data guide](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/docs/Train_custom_data.md) and the [COCO training guide](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/docs/Train_coco_data.md) before running those workflows.

## What I learned

### About learning algorithms

- A weight is a parameter that changes to reduce prediction error.
- A bias lets a model shift its predictions instead of forcing every line through zero.
- The learning rate controls the size of each optimization step.
- Regression predicts a continuous quantity; classification predicts a class or probability.
- Logistic regression is a probability model whose threshold determines the final label.
- Decision trees are explainable but need depth control to avoid memorization.
- Random forests reduce the variance of individual trees by averaging many different trees.
- KNN depends heavily on feature scaling because it compares distances directly.
- SVMs define boundaries using support vectors and can model non-linear patterns with kernels.
- Naive Bayes is fast, but its feature-independence assumption must be considered.
- Accuracy is not enough when the important class is rare or when missed failures are costly.
- Confusion matrices make the type of mistake visible instead of hiding it inside one score.

### About computer vision

- Object detection combines classification with localization.
- Confidence thresholds control how many candidate detections are kept.
- Non-maximum suppression removes duplicate boxes around the same object.
- Image resizing and letterboxing affect both inference speed and box coordinates.
- Data augmentation helps a detector see more variation during training.
- Evaluation must consider localization quality, not only whether an object class was guessed.
- A useful model workflow continues beyond training: inference, export, runtime testing, and deployment matter too.
- Quantization can improve deployment efficiency, but accuracy must be checked after conversion.

### Internship working method

- Research the mathematical idea before coding.
- Build a small example where each intermediate value can be printed.
- Use a visualization to inspect what the model learned.
- Compare training behavior with held-out evaluation.
- Keep rough experiments because they show how the final understanding developed.
- Document assumptions, limitations, and practical trade-offs alongside the code.

## Resources

- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [scikit-learn model evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [YOLOv6 local documentation](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/README.md)
- [YOLOv6 training guide](YOLO%20family/YOLO%20%201%20-%2026%20models/YOLOv6/docs/Train_custom_data.md)
- StatQuest videos for model intuition
- 3Blue1Brown videos for optimization and mathematical intuition
- NotebookLM for research, comparisons, and implementation notes
- VS Code and Jupyter for execution, debugging, and visualization

## Notes

- The ML examples use synthetic educational data and are not production fleet-health models.
- YOLO notebooks may depend on external packages, checkpoints, datasets, or runtime-specific hardware.
- The YOLOv6 directory contains a substantial implementation and its upstream technical documentation; the root README explains how it fits into this internship project.
