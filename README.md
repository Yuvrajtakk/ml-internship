# ML Internship Learning Journal

**Yuvraj Tak** | B.Tech CSE-AI (2023–2027) | Anand International College of Engineering, Jaipur

**Remote AI/ML Internship** at Watsoo Express Pvt. Ltd.
**Mentor:** Ankit Gupta

## Project overview

This repository is my complete documented learning journey for a 2-month AI/ML internship.
It was built as a single-threaded path from first principles to practical model evaluation:
- start with the meaning of a weight, error, learning rate, SGD, and bias
- build each algorithm in a dedicated file
- use visuals to make the results easy to understand
- keep Watsoo Express truck data as the repeated theme for classification

The code lives under `Understanding_and_Implementation/CODE BOOK/` and the rough experimental work and output figures are in `Understanding_and_Implementation/ROUGH CODE/`.

## The learning journey

1. **Foundation first**
   - I began by coding ML from scratch in `00_foundation.py`, without `sklearn`.
   - The goal was to understand the exact loop that every learning algorithm uses: predict, measure error, update weights, repeat.
2. **Learning plan**
   - I created a structured plan using a curated playlist of StatQuest and 3Blue1Brown videos.
   - I matched each algorithm to the official `scikit-learn` documentation and organized a to-do list for the project.
3. **Deep research with NotebookLM**
   - NotebookLM helped me compile theoretical frameworks, compare algorithms, and gather precise implementation references.
   - This made the code more than just copy-paste: it became a reasoned implementation backed by research.
4. **Iterative learning**
   - For each algorithm: math first, then code, then visualization.
   - Each file became a standalone lesson.
5. **Final evaluation**
   - I wrapped the journey with `08_evaluation_metrics.py`, where I showed why accuracy can lie and why precision, recall, and F1 matter.

## Repository structure

- `Understanding_and_Implementation/CODE BOOK/`
  - `00 foundation.py`
  - `01_linear_regression.py`
  - `02_logistic_regression.py`
  - `03_decision_tree.py`
  - `04_random_forest.py`
  - `05_knn.py`
  - `06_svm.py`
  - `07_naive_bayes.py`
  - `08_evaluation_metrics.py`
- `Understanding_and_Implementation/ROUGH CODE/`
  - rough experiments and additional output images

## File walkthrough

### `00_foundation.py`

What it covers:
- the core learning loop in pure Python
- one-point learning and weight updates
- learning rate impact and epochs
- SGD across multiple data points
- bias/intercept and why it matters
- a visualization of error dropping and the learned line

Key concepts:
- weight = model parameter
- prediction = input × weight
- error = target − prediction
- learning rate controls update size
- bias allows shifting the line away from origin

Code approach:
- start with one input and one weight
- then expand to multiple points and update weights after each example
- finally add bias and visualize the learned regression line and loss curve

Output:

![](Understanding_and_Implementation/CODE%20BOOK/Fundamenrtal.png)

### `01_linear_regression.py`

What it covers:
- ordinary least squares linear regression with `sklearn`
- synthetic truck fuel-cost dataset for Watsoo Express
- train/test split, RMSE, and R² score
- predicted cost for a new truck
- scatter & regression-line visualization

Key concepts:
- regression predicts a continuous quantity
- slope and intercept define the learned formula
- noise simulates real-world sensor/driver variation
- R² measures how well the model explains variance

Code approach:
- generate 80 synthetic trucks with distance and fuel cost
- train a `LinearRegression` model on 80% of the data
- evaluate on 20% of trucks held out from training
- plot the learned line and actual vs predicted values

Output:

![](Understanding_and_Implementation/CODE%20BOOK/linear_regresion.png)

### `02_logistic_regression.py`

What it covers:
- binary classification using logistic regression
- a truck health dataset with speed and vibration features
- standard scaling before training
- accuracy, precision, recall, and F1 score
- threshold tuning from 0.5 to 0.3
- sigmoid curve, decision boundary, and threshold trade-off

Key concepts:
- logistic regression outputs probabilities, not labels
- sigmoid squashes linear output into [0, 1]
- decision threshold controls sensitivity vs false alarms
- precision is for false alarm cost, recall is for missed failures

Code approach:
- build healthy vs failing truck clusters in feature space
- scale features for stable optimization
- fit `LogisticRegression` and inspect predictions
- demonstrate how a lower threshold catches more failures

Output:

![](Understanding_and_Implementation/CODE%20BOOK/Logistic_Regresion.png)

### `03_decision_tree.py`

What it covers:
- decision tree classification using `DecisionTreeClassifier`
- Gini impurity and how splits are selected
- controlled depth vs overfitting demonstration
- feature importance and human-readable rules
- decision boundary visualization and flowchart

Key concepts:
- trees split data with yes/no questions
- shallow trees generalize better than deep trees
- overfitting = high train accuracy, low test accuracy
- trees do not require scaling

Code approach:
- generate trucks with engine temperature, vibration, and oil pressure
- train a depth-constrained tree and a fully grown tree
- compare their train/test performance
- plot the learned decision regions and the tree structure

Output:

![](Understanding_and_Implementation/CODE%20BOOK/Decision%20Tree.png)

### `04_random_forest.py`

What it covers:
- random forest ensembles of decision trees
- bootstrap sampling, feature randomness, and majority vote
- out-of-bag (OOB) validation score
- comparison to a single decision tree
- feature importances and ensemble decision region

Key concepts:
- ensemble learning reduces variance
- each tree sees a random subset of rows and features
- OOB score is a built-in validation estimate
- forests generalize better than a single deep tree

Code approach:
- build healthy, failing, and ambiguous truck examples
- train both a single tree and a 100-tree forest
- compare train vs test performance and OOB accuracy
- visualize decision boundaries and sample trees

Output:

![](Understanding_and_Implementation/CODE%20BOOK/Rabdon%20Forest.png)

### `05_knn.py`

What it covers:
- K-Nearest Neighbors classification
- why scaling is mandatory for distance-based models
- the U-curve for choosing K
- K=1, K=5, K=11 comparisons
- boundary visualizations for scaled vs unscaled data

Key concepts:
- KNN stores the training set and predicts by vote
- distance dominates when features have different scales
- small K → overfitting, large K → underfitting
- odd K avoids tie votes in binary classification

Code approach:
- create truck clusters with speed and vibration
- scale the data, compare raw and scaled boundaries
- use cross-validation to find the best K
- visualize how the decision boundary changes with K

Output:

![](Understanding_and_Implementation/CODE%20BOOK/KNN%20implementation.png)

### `06_svm.py`

What it covers:
- Support Vector Machine with 3 kernels: Linear, RBF, Polynomial
- support vectors and maximum-margin intuition
- kernel trick for non-linear decision boundaries
- small synthetic datasets designed for each kernel
- visualization of boundaries and support vectors

Key concepts:
- SVM focuses on the closest points, not all points
- linear kernel fits a straight hyperplane
- RBF kernel handles circular/blob-shaped separation
- polynomial kernel fits curved moon-shaped separation

Code approach:
- generate three datasets matched to each kernel
- scale all features first
- fit `SVC` models and report accuracy and support vector counts
- visualize the margin and decision regions

Output:

![](Understanding_and_Implementation/CODE%20BOOK/SVM.png)

### `07_naive_bayes.py`

What it covers:
- three Naive Bayes variants: Gaussian, Multinomial, Bernoulli
- Bayes' theorem and the independence assumption
- continuous sensor data, word counts, and binary checklists
- probability interpretation for each variant
- example predictions for new truck observations

Key concepts:
- GaussianNB models continuous features with bell curves
- MultinomialNB models integer counts like words or event frequencies
- BernoulliNB models binary pass/fail checklist data
- Naive Bayes is fast and works well on small datasets

Code approach:
- build synthetic datasets for each assumption type
- fit each `sklearn` Naive Bayes variant
- print learned statistics and class probabilities
- visualize decision regions and class-conditional probabilities

Output:

![](Understanding_and_Implementation/CODE%20BOOK/Naive%20Bayes.png)

### `08_evaluation_metrics.py`

What it covers:
- the accuracy trap on imbalanced data
- manual TP/TN/FP/FN walkthrough
- logistic regression performance on a real-like truck dataset
- accuracy, precision, recall, and F1 score comparisons
- threshold trade-off and classification report

Key concepts:
- accuracy can be misleading when classes are imbalanced
- precision penalizes false alarms, recall penalizes missed failures
- F1 score balances precision and recall
- confusion matrices make errors visible

Code approach:
- show a lazy model that predicts only the majority class
- calculate metrics by hand and with `sklearn`
- train a logistic regression model and compare metric thresholds
- visualize the confusion matrix and metric trade-offs

Output:

![](Understanding_and_Implementation/CODE%20BOOK/EVALUATION%20METRICS.png)

## Key learnings

- The same core learning loop underlies all models: predict, measure error, update.
- A model's weights are not magic; they are parameters that move to reduce error.
- Bias is essential to shift predictions away from the origin.
- Regression and classification are different goals: numbers vs categories.
- Logistic regression is really a probability model with a decision threshold.
- Decision trees are explainable, but can overfit without depth control.
- Random forests turn many weak trees into a stronger, more stable learner.
- KNN is powerful but only after proper feature scaling and K selection.
- SVM learns boundaries from support vectors and can use kernels for complex shapes.
- Naive Bayes is fast and surprisingly effective even under strong independence assumptions.
- Accuracy is not enough; precision, recall, and F1 are critical for real-world safety problems.

## How to run

### Requirements

- Python 3.8+ recommended
- `numpy`
- `matplotlib`
- `scikit-learn`

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy matplotlib scikit-learn
```

### Run the examples

From the repository root:

```bash
python "Understanding_and_Implementation/CODE BOOK/00 foundation.py"
python "Understanding_and_Implementation/CODE BOOK/01_linear_regression.py"
python "Understanding_and_Implementation/CODE BOOK/02_logistic_regression.py"
python "Understanding_and_Implementation/CODE BOOK/03_decision_tree.py"
python "Understanding_and_Implementation/CODE BOOK/04_random_forest.py"
python "Understanding_and_Implementation/CODE BOOK/05_knn.py"
python "Understanding_and_Implementation/CODE BOOK/06_svm.py"
python "Understanding_and_Implementation/CODE BOOK/07_naive_bayes.py"
python "Understanding_and_Implementation/CODE BOOK/08_evaluation_metrics.py"
```

Each script opens visualizations and prints results to the terminal.

## Resources used

- StatQuest videos for each algorithm and model intuition
- 3Blue1Brown videos for deep conceptual understanding of learning curves and functions
- `scikit-learn` documentation for:
  - linear regression: https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
  - logistic regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  - decision trees: https://scikit-learn.org/stable/modules/tree.html
  - random forests: https://scikit-learn.org/stable/modules/ensemble.html#forest
  - KNN: https://scikit-learn.org/stable/modules/neighbors.html#classification
  - SVM: https://scikit-learn.org/stable/modules/svm.html
  - Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
  - model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- NotebookLM for research, algorithm comparison, and implementation reference
- VS Code for code execution, debugging, and visualization

## Notes

- The `ROUGH CODE` folder contains early experiments, extra output visuals, and exploratory notebooks.
- This repository is intentionally built as a documented journal, not just a finished model.
- The goal was to learn and explain each step, not just get a single score.
