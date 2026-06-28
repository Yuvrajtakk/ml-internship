# # 1. Our data: We know an input of 2 should result in an output of 6.
# input_data = 2          #Feature
# target_output = 6       #Label

# # 2. Our machine's internal "weight" (starting with a random guess)
# w = 1.0

# # 3. The machine makes a prediction using its current weight
# prediction = input_data * w

# # 4. We calculate how wrong the machine is
# error = target_output - prediction

# print("Prediction:", prediction)
# print("Error:", error)

# # 1. The learning rate controls how big of a step the machine takes
# learning_rate = 2.0

# # 2. A loop to let the machine practice 10 times (10 epochs)
# for epoch in range(10):
    
#     # 3. Make a prediction and calculate the error
#     prediction = input_data * w
#     error = target_output - prediction
    
#     # 4. The "Learning" step: adjusting the weight
#     w = w + (learning_rate * error * input_data)
    
#     # 5. Print the results for this epoch
#     print(f"Epoch {epoch+1}: Prediction = {prediction:.2f}, Error = {error:.2f}, New Weight = {w:.2f}")


# # 1. Python Lists: Using square brackets to hold multiple data points - Python lists
# inputs = [1, 2, 3, 4]
# targets = [3, 6, 9, 12]

# w = 1.0
# learning_rate = 0.05 

# # 2. Outer loop: The number of times we pass through the ENTIRE dataset (Epochs)
# for epoch in range(5):
#     print(f"\n--- Epoch {epoch + 1} ---")
    
#     # 3. Inner loop: Iterating through the dataset one pair at a time
#     for x, y in zip(inputs, targets): #zip() function acts like physical zipper, pairing up first item of inputs with first item of targets, second with second, and so on.
        
#         # 4. Make a prediction using the current input (x)
#         prediction = x * w
        
#         # 5. Calculate the error using the current target (y)
#         error = y - prediction
        
#         # 6. Update the weight immediately based on this specific error
#         w = w + (learning_rate * error * x)
        
#         print(f"Input: {x}, Target: {y} | Pred: {prediction:.2f}, Error: {error:.2f} | New Weight: {w:.2f}")

# # 1. New dataset: We are changing the underlying rule
# inputs = [1, 2, 3, 4]
# targets = [5, 7, 9, 11]

# # 2. We now have TWO internal parameters to learn
# w = 1.0  # Weight
# b = 0.0  # Bias

# learning_rate = 0.05 

# # We increased the epochs to 15 to give it more time to learn two variables
# for epoch in range(15):
#     print(f"\n--- Epoch {epoch + 1} ---")
    
#     for x, y in zip(inputs, targets):
        
#         # 3. The prediction now includes the bias
#         prediction = (x * w) + b
        
#         error = y - prediction
        
#         # 4. We must update BOTH parameters based on the same error. 
#         w = w + (learning_rate * error * x)
#         b = b + (learning_rate * error * 1) 
        
#         print(f"In: {x}, Target: {y} | Pred: {prediction:.2f}, Error: {error:.2f} | w: {w:.2f}, b: {b:.2f}")


# inputs = [1, 2, 3, 4]
# targets = [5, 7, 9, 11]

# w = 1.0  
# b = 0.0  
# learning_rate = 0.05 

# # 1. Increased to 200 epochs to give the machine more time
# for epoch in range(200):
#     for x, y in zip(inputs, targets):
        
#         prediction = (x * w) + b
#         error = y - prediction
        
#         w = w + (learning_rate * error * x)
#         b = b + (learning_rate * error * 1) 
        
#     # 2. Only print the results every 20 epochs to keep the terminal clean
#     if (epoch + 1) % 20 == 0:
#         print(f"Epoch {epoch + 1} | w: {w:.2f}, b: {b:.2f} | Final Error: {error:.2f}")


# import matplotlib.pyplot as plt

# inputs = [1, 2, 3, 4]
# targets = [5, 7, 9, 11]

# # The final parameters our machine learned
# w = 2.0
# b = 3.0

# # 1. List Comprehension: A fast, Pythonic way to run a loop inside a list
# # This calculates the predicted y value for every x in our inputs
# predictions = [(x * w) + b for x in inputs]

# # 2. Scatter plot: Draws our original dataset as individual dots
# plt.scatter(inputs, targets, color='blue', label='Actual Data')

# # 3. Line plot: Draws our machine's learned rule as a continuous line
# plt.plot(inputs, predictions, color='red', label='Learned Rule')

# # 4. Make the graph readable
# plt.xlabel('Inputs (x)')
# plt.ylabel('Targets (y)')
# plt.title('Machine Learning: Fitting a Line to Data')
# plt.legend()

# # 5. Render the window on your screen
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # 1. Define our squashing function
# def sigmoid(z):
#     # np.exp(-z) is numpy's way of calculating e^(-z)
#     return 1 / (1 + np.exp(-z))

# # 2. Create some dummy data representing the output of a linear equation
# # np.linspace creates an array of exactly 100 evenly spaced numbers between -10 and 10
# linear_outputs = np.linspace(-10, 10, 100)

# # 3. Pass our linear outputs through the sigmoid function
# probabilities = sigmoid(linear_outputs)

# # 4. Visualize the result
# plt.plot(linear_outputs, probabilities, color='purple', linewidth=3)
# plt.axhline(0.5, color='gray', linestyle='--') # Draws a dashed line at 0.5
# plt.title('The Sigmoid Function')
# plt.xlabel('Linear Output (z = wx + b)')
# plt.ylabel('Probability (squashed between 0 and 1)')
# plt.grid(True)
# plt.show()

# import numpy as np
# from sklearn.linear_model import LogisticRegression

# inputs = np.array([[1], [2], [3], [4], [5], [6]])
# targets = np.array([0, 0, 0, 1, 1, 1])

# model = LogisticRegression()

# model.fit(inputs, targets)

# new_data = np.array([[3.5]])
# prediction = model.predict(new_data)
# probability = model.predict_proba(new_data)

# print(f"Prediction: {prediction[0]}")
# print(f"Probabilities: Class 0 = {probability[0][0]:.2f}, Class 1 = {probability[0][1]:.2f}")

# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, precision_score, recall_score

# # 1. Dummy data: 10 engines. (0 = Healthy, 1 = Failing)
# # Let's say feature is "Engine Temperature"
# temperatures = np.array([[200], [210], [220], [230], [240], [250], [260], [270], [280], [290]])
# actual_status = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

# # 2. Train the model
# model = LogisticRegression()
# model.fit(temperatures, actual_status)

# # 3. Get the raw probabilities instead of the default 0.5 predictions
# probabilities = model.predict_proba(temperatures)

# # 4. Extract just the probability of class 1 (Failing)
# # probabilities is a 2D array: [prob_class_0, prob_class_1]. We want the second column.
# prob_failing = probabilities[:, 1] 

# # 5. Apply a CUSTOM "paranoid" threshold of 0.3
# custom_predictions = (prob_failing >= 0.3).astype(int)

# # 6. Calculate our metrics
# matrix = confusion_matrix(actual_status, custom_predictions)
# recall = recall_score(actual_status, custom_predictions)
# precision = precision_score(actual_status, custom_predictions)

# print("Confusion Matrix:\n", matrix)
# print(f"Recall: {recall:.2f}")
# print(f"Precision: {precision:.2f}")


# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.datasets import load_iris

# # 1. Load a sample dataset (Iris)
# iris = load_iris()
# X, y = iris.data, iris.target

# # 2. Create the blueprint for the tree
# # We set max_depth to 3 to keep the tree small and readable
# clf = DecisionTreeClassifier(max_depth=3)

# # 3. Fit the model to the data
# clf.fit(X, y)

# # 4. Make a prediction for a new, unseen flower
# # Features: [sepal length, sepal width, petal length, petal width]
# new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
# prediction = clf.predict(new_flower)

# print(f"Prediction: {iris.target_names[prediction][0]}")


# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier, plot_tree

# # 1. Our tiny dataset (2D array for features, 1D for targets)
# X = [[10], [20], [30], [40]]
# y = [0, 0, 1, 1]

# # 2. Instantiate the model
# tree_model = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=42)

# # 3. Train the model
# tree_model.fit(X, y)

# # 4. Predict a new unseen temperature (e.g., 18 degrees)
# new_temp = [[18]]
# prediction = tree_model.predict(new_temp)
# print(f"Prediction for Temp 18: {prediction[0]}")

# # 5. Visualize the math the tree just performed
# plt.figure(figsize=(8, 6))
# plot_tree(tree_model, feature_names=["Temperature"], class_names=["Healthy", "Failing"], filled=True)
# plt.title("Decision Tree: Gini Impurity in Action")
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import plot_tree

# X = np.array([
#     [200, 0.1, 10],
#     [210, 0.2, 11],
#     [220, 0.1, 10],
#     [280, 0.8, 15],
#     [290, 0.9, 14],
#     [300, 0.8, 16]
# ])
# y = np.array([0, 0, 0, 1, 1, 1])

# rf_model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
# rf_model.fit(X, y)

# new_engine = np.array([[250, 0.5, 12]])
# prediction = rf_model.predict(new_engine)
# probabilities = rf_model.predict_proba(new_engine)

# print(f"Prediction: {prediction[0]}")
# print(f"Voting Results: Healthy = {probabilities[0][0]*100}%, Failing = {probabilities[0][1]*100}%")

# # Visualize first 6 trees from the forest
# fig, axes = plt.subplots(2, 3, figsize=(20, 10))
# feature_names = ["Temperature", "Vibration", "Pressure"]
# class_names = ["Healthy", "Failing"]

# for i, ax in enumerate(axes.ravel()):
#     plot_tree(
#         rf_model.estimators_[i],
#         feature_names=feature_names,
#         class_names=class_names,
#         filled=True,
#         ax=ax
#     )
#     ax.set_title(f"Tree {i+1}", fontsize=12)

# plt.suptitle("6 Individual Trees Inside the Random Forest", fontsize=16)
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier

# # 1. Synthetic 2D dataset: [Speed, Vibration]
# X = np.array([
#     [40, 0.2], [45, 0.1], [50, 0.3], [55, 0.2], # Healthy trucks (0)
#     [85, 0.8], [90, 0.9], [95, 0.7], [100, 0.8], # Failing trucks (1)
#     [70, 0.4], [75, 0.6]                          # Overlapping / Tricky zone
# ])
# y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])

# # 2. Initialize the Random Forest with OOB scoring enabled
# forest = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=42)

# # 3. Train the model
# forest.fit(X, y)

# # 4. Create a dense grid grid to paint the decision boundary background
# x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
# y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

# # Generate coordinates for every intersection on our grid mesh
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),np.arange(y_min, y_max, 0.02))

# # 5. Predict the class for every single point on the background grid
# # np.c_ flattens the grids and pairs them up as [x, y] coordinates
# grid_points = np.c_[xx.ravel(), yy.ravel()]
# Z = forest.predict(grid_points)
# Z = Z.reshape(xx.shape) # Shape it back into a 2D grid image

# # 6. Plot the boundary background and coordinates
# plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr') # background color fill
# plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', label='Healthy (0)')
# plt.scatter(X[y==1, 0], X[y==1, 1], color='red', label='Failing (1)')

# plt.title(f"Random Forest Decision Boundary | OOB Accuracy: {forest.oob_score_:.2%}")
# plt.xlabel('Truck Speed')
# plt.ylabel('Vibration Level')
# plt.legend()
# plt.show()


# #python:Decision Tree Space Splitter:decision_tree_walkthrough.py
# import numpy as np  # We import NumPy to handle arrays and fast grid mathematical computations.
# import matplotlib.pyplot as plt  # We import Matplotlib's pyplot interface to draw coordinates and fill backgrounds.
# from sklearn.tree import DecisionTreeClassifier, plot_tree  # We import the classifier model and its built-in visualization.

# # 1. Prepare our training dataset (X is 2D features, y is 1D binary labels).
# # Feature columns: [Temperature (Celsius), Vibration Level (G-force)].
# X = np.array([
#     [12.0, 0.15],  # Engine 1: Temp=12, Vib=0.15 -> Class 0 (Healthy)
#     [18.0, 0.22],  # Engine 2: Temp=18, Vib=0.22 -> Class 0 (Healthy)
#     [22.0, 0.65],  # Engine 3: Temp=22, Vib=0.65 -> Class 1 (Failing)
#     [32.0, 0.78],  # Engine 4: Temp=32, Vib=0.78 -> Class 1 (Failing)
#     [35.0, 0.35]   # Engine 5: Temp=35, Vib=0.35 -> Class 0 (Healthy)
# ])
# y = np.array([0, 0, 1, 1, 0])  # status targets: 0 represents Healthy, 1 represents Failing.

# # 2. Instantiate the Decision Tree model.
# # max_depth=2 stops tree expansion after 2 layers to prevent overfitting on this small set.
# # criterion='gini' specifies that we use the Gini Impurity metric we calculated manually.
# tree_model = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=42)

# # 3. Fit the model to the input features X and target classes y.
# # During fit, the tree recursively sweeps through features and thresholds to find the highest Gini reduction.
# tree_model.fit(X, y)

# # 4. Create a Matplotlib figure containing two subplots (1 row, 2 columns).
# # figsize=(14, 6) sets the physical width of the popped window to 14 inches and height to 6 inches.
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# # --- FIRST SUBPLOT: Drawing the decision boundary space map ---
# # We find the min and max limits of our features, and pad them to make space for the boundaries.
# x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5  # Limits for temperature axis.
# y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1  # Limits for vibration axis.

# # np.meshgrid creates a dense grid of coordinates representing every pixel in the graph coordinate system.
# # np.arange(start, stop, step) generates values at small intervals to make the boundary background smooth.
# xx, yy = np.meshgrid(
#     np.arange(x_min, x_max, 0.1),  # Temperature values with a step of 0.1
#     np.arange(y_min, y_max, 0.01)  # Vibration values with a step of 0.01
# )

# # We flatten the grids and pair up each (x, y) intersection to feed into the prediction engine.
# # xx.ravel() flattens the 2D grid matrix into a 1D vector.
# # np.c_ takes these flattened vectors and merges them side-by-side as coordinate pairs.
# grid_pixels = np.c_[xx.ravel(), yy.ravel()]

# # Predict the class (0 or 1) for every single coordinate pixel on the graph background.
# pixel_predictions = tree_model.predict(grid_pixels)

# # Reshape the output vector of predictions back into a 2D grid to match the structure of our coordinates.
# pixel_predictions = pixel_predictions.reshape(xx.shape)

# # ax1.contourf fills the background of the first subplot with color maps based on our predictions.
# # alpha=0.3 makes the background semi-transparent; cmap='bwr' maps class 0 to blue and class 1 to red.
# ax1.contourf(xx, yy, pixel_predictions, alpha=0.3, cmap='bwr')

# # Scatter plot our original 5 points on top of the shaded decision boundaries.
# # We plot the Healthy engines (y == 0) as blue markers and Failing engines (y == 1) as red markers.
# ax1.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', edgecolor='k', s=100, label='Healthy (0)')
# ax1.scatter(X[y == 1, 0], X[y == 1, 1], color='red', edgecolor='k', s=100, label='Failing (1)')

# # Configure labels, titles, and legends for the coordinate map.
# ax1.set_xlabel('Temperature (°C)')  # Label for the horizontal x-axis.
# ax1.set_ylabel('Vibration Level (G-force)')  # Label for the vertical y-axis.
# ax1.set_title('Decision Tree: Carved Space Boundaries')  # Subplot title.
# ax1.legend()  # Display our labeled dataset items.

# # --- SECOND SUBPLOT: Plotting the logical flowchart tree ---
# # plot_tree draws the flowchart representation of our nested rules.
# # filled=True colors the flowchart blocks to represent which class is dominant.
# plot_tree(
#     tree_model,
#     feature_names=["Temperature", "Vibration"],
#     class_names=["Healthy", "Failing"],
#     filled=True,
#     ax=ax2
# )
# ax2.set_title('Decision Tree: Logical Decision Rules')  # Subplot title.

# # Display the finalized multi-pane visualization on your screen.
# plt.tight_layout()  # Adjusts subplot padding so labels do not overlap.
# plt.show()  # Opens the interactive display window in VS Code.


# #python:Random Forest Ensemble Mapping:random_forest_walkthrough.py
# import numpy as np  # We import NumPy to handle fast coordinate vectors and matrix slicing operations.
# import matplotlib.pyplot as plt  # We import Matplotlib's pyplot to render decision territory maps.
# from sklearn.ensemble import RandomForestClassifier  # We import the Random Forest ensemble model from scikit-learn.

# # 1. Generate synthetic 2D data: [Speed (km/h), Fuel Consumption Rate (L/100km)].
# # This dataset contains some overlapping boundary points to show how the ensemble handles noisy regions.
# X = np.array([
#     [30, 5.5], [35, 6.2], [40, 5.8], [45, 6.0], [50, 7.2],  # Healthy delivery trucks (0)
#     [85, 12.5], [90, 14.1], [95, 13.8], [100, 15.0], [75, 11.2],  # Damaged/Faulty delivery trucks (1)
#     [60, 8.5], [65, 10.1], [70, 9.2], [72, 8.8]  # Ambiguous middle-ground trucks
# ])
# y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1])  # status targets: 0=Healthy, 1=Failing.

# # 2. Instantiate our Random Forest model.
# # n_estimators=100 specifies that we will build an ensemble of exactly 100 decision trees.
# # oob_score=True forces the forest to calculate the out-of-bag validation accuracy using the 36.8% left-out data.
# # max_features='sqrt' restricts each tree to split using a maximum of sqrt(total_features) variables.
# forest_model = RandomForestClassifier(n_estimators=100, oob_score=True, max_features='sqrt', random_state=42)

# # 3. Train the model using fit.
# # This bootstrap-samples the rows and feature indices 100 times to construct our collection of trees.
# forest_model.fit(X, y)

# # 4. Generate coordinates to plot our decision boundaries.
# x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10  # Horizontal boundary range for speed.
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Vertical boundary range for fuel rate.

# # Generate the grid mesh intersections.
# xx, yy = np.meshgrid(
#     np.arange(x_min, x_max, 0.5),  # Step size of 0.5 for speed.
#     np.arange(y_min, y_max, 0.05)  # Step size of 0.05 for fuel consumption.
# )

# # Flatten and stack coordinate lists.
# grid_coordinates = np.c_[xx.ravel(), yy.ravel()]

# # Predict class labels for every point on our grid background.
# grid_predictions = forest_model.predict(grid_coordinates)

# # Reshape prediction output matrix to match the spatial geometry of our grid.
# grid_predictions = grid_predictions.reshape(xx.shape)

# # Create our visualization canvas window.
# plt.figure(figsize=(10, 7))

# # Contour fill our map area with color bands.
# # Class 0 regions are colored light blue; Class 1 regions are colored light red.
# plt.contourf(xx, yy, grid_predictions, alpha=0.3, cmap='bwr')

# # Draw our original truck points on top of the prediction map.
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', edgecolor='k', s=100, label='Healthy Truck (0)')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', edgecolor='k', s=100, label='Failing Truck (1)')

# # Configure axes titles, labels, grids, and legends.
# plt.xlabel('Truck Speed (km/h)')  # Label our horizontal coordinate axis.
# plt.ylabel('Fuel Consumption Rate (L/100km)')  # Label our vertical coordinate axis.

# # We pull our calculated out-of-bag validation metric score using the forest_model.oob_score_ attribute.
# plt.title(f"Random Forest (100 Trees) | Out-Of-Bag Validation Accuracy: {forest_model.oob_score_:.2%}")
# plt.legend()  # Display labels.
# plt.grid(True, linestyle='--', alpha=0.5)  # Overlay a soft grid line system.

# # Draw the window screen.
# plt.show()



# #python:Support Vector Machine Margin Visualizer:svm_margin_walkthrough.py
# import numpy as np  # Used to construct coordinate vectors and matrix spacing arrays
# import matplotlib.pyplot as plt  # Used to render our decision space maps and dot plots
# from sklearn.svm import SVC  # Used to import the standard support vector binary classification machine

# # 1. Generate an array grid of 8 coordinate positions representing truck parameters [Speed, Vibrations]
# X = np.array([
#     [2, 3], [3, 2], [1, 1], [4, 1],  # Healthy delivery trucks (Class 0)
#     [6, 7], [7, 6], [8, 8], [5, 9]   # Failing delivery trucks (Class 1)
# ])
# y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Target classifications mapped to each truck vector

# # 2. Instantiate and fit a linear linear-kernel SVM machine
# # C=1.0 specifies our strictness penalty factor; kernel='linear' locks the algorithm to straight separating hyperplanes
# svm_model = SVC(kernel='linear', C=1.0)
# svm_model.fit(X, y)  # Executes convex quadratic programming loops to mathematically isolate support points

# # 3. Establish structural figure layout elements using matplotlib
# plt.figure(figsize=(10, 7))  # Creates a layout sheet sized 10 inches wide by 7 inches tall

# # 4. Generate coordinate mesh grids to construct our continuous color-filled boundary background
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Border calculations tracking the Speed feature limits
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Border calculations tracking the Vibration feature limits
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))  # Build coordinate mesh sheets

# # 5. Calculate class predictions for every specific mesh intersection pixel on our visual background sheet
# Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)  # Reshape vector sheets back into data matrices
# plt.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')  # Render our divided filled regional background colors (Blue/Red)

# # 6. Isolate and draw the mathematical decision hyperplane line and its corresponding margin boundaries
# w = svm_model.coef_[0]  # Extract the isolated weight parameter vector [w0, w1] calculated by the optimizer
# slope = -w[0] / w[1]  # Derive standard geometric straight-line slope values from structural vector weights
# intercept = -svm_model.intercept_[0] / w[1]  # Scale coordinate intercept offsets relative to vertical dimension maps
# axis_range = np.linspace(x_min, x_max, 100)  # Continuous linear value space tracking our horizontal scale range
# hyperplane_line = slope * axis_range + intercept  # Calculate standard line point positions tracking the main hyperplane
# plt.plot(axis_range, hyperplane_line, 'k-', linewidth=2, label='Separating Hyperplane')  # Plot solid line center boundary

# # 7. Calculate and overlay parallel margin lanes derived from support vector coordinate spacing constraints
# margin_offset = 1 / w[1]  # Scale vertical line spacing gaps using extracted weight norm projections
# margin_above = hyperplane_line + margin_offset  # Shift baseline points vertically to mark upper class bounds
# margin_below = hyperplane_line - margin_offset  # Shift baseline points vertically to mark lower class bounds
# plt.plot(axis_range, margin_above, 'k--', linewidth=1.5, label='Margin Boundaries')  # Plot upper dashed pavement track
# plt.plot(axis_range, margin_below, 'k--', linewidth=1.5)  # Plot lower dashed pavement track

# # 8. Circle specific points saved in internal model state memory arrays identified as Support Vectors
# support_points = svm_model.support_vectors_  # Extract coordinate values saved inside internal model storage fields
# plt.scatter(support_points[:, 0], support_points[:, 1], s=250, facecolors='none', edgecolors='black', linewidths=2.5, label='Support Vectors')  # Ring vectors

# # 9. Scatter plot original point distributions grouped by target labels over the shaded territory spaces
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', edgecolor='k', s=100, label='Healthy Truck (0)')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', edgecolor='k', s=100, label='Failing Truck (1)')

# # 10. Finalize descriptive chart legend titles, labeled axes elements, and screen window layouts
# plt.title('Support Vector Machine: Maximum Margin Space Separation')  # Write structural map heading titles
# plt.xlabel('Truck Speed (Normalized Scale)')  # Label horizontal feature orientation vectors
# plt.ylabel('Vibration Energy (Normalized Scale)')  # Label vertical feature orientation vectors
# plt.legend(loc='upper left')  # Draw bounded identifying descriptive map keys
# plt.grid(True, linestyle=':', alpha=0.6)  # Layer lightweight structural coordinate alignment dots
# plt.show()  # Command standard operating kernels to project finalized map structures onto user screens



#python:Support Vector Machine Margin Visualizer:svm_margin_walkthrough.py
import numpy as np  # Used to construct coordinate vectors and matrix spacing arrays
import matplotlib.pyplot as plt  # Used to render our decision space maps and dot plots
from sklearn.svm import SVC  # Used to import the standard support vector binary classification machine

# 1. Generate an array grid of 8 coordinate positions representing truck parameters [Speed, Vibrations]
X = np.array([
    [2, 3], [3, 2], [1, 1], [4, 1],  # Healthy delivery trucks (Class 0)
    [6, 7], [7, 6], [8, 8], [5, 9]   # Failing delivery trucks (Class 1)
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Target classifications mapped to each truck vector

# 2. Instantiate and fit a linear linear-kernel SVM machine
# C=1.0 specifies our strictness penalty factor; kernel='linear' locks the algorithm to straight separating hyperplanes
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X, y)  # Executes convex quadratic programming loops to mathematically isolate support points

# 3. Establish structural figure layout elements using matplotlib
plt.figure(figsize=(10, 7))  # Creates a layout sheet sized 10 inches wide by 7 inches tall

# 4. Generate coordinate mesh grids to construct our continuous color-filled boundary background
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Border calculations tracking the Speed feature limits
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Border calculations tracking the Vibration feature limits
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))  # Build coordinate mesh sheets

# 5. Calculate class predictions for every specific mesh intersection pixel on our visual background sheet
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)  # Reshape vector sheets back into data matrices
plt.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')  # Render our divided filled regional background colors (Blue/Red)

# 6. Isolate and draw the mathematical decision hyperplane line and its corresponding margin boundaries
w = svm_model.coef_[0]  # Extract the isolated weight parameter vector [w0, w1] calculated by the optimizer
slope = -w[0] / w[1]  # Derive standard geometric straight-line slope values from structural vector weights
intercept = -svm_model.intercept_[0] / w[1]  # Scale coordinate intercept offsets relative to vertical dimension maps
axis_range = np.linspace(x_min, x_max, 100)  # Continuous linear value space tracking our horizontal scale range
hyperplane_line = slope * axis_range + intercept  # Calculate standard line point positions tracking the main hyperplane
plt.plot(axis_range, hyperplane_line, 'k-', linewidth=2, label='Separating Hyperplane')  # Plot solid line center boundary

# 7. Calculate and overlay parallel margin lanes derived from support vector coordinate spacing constraints
margin_offset = 1 / w[1]  # Scale vertical line spacing gaps using extracted weight norm projections
margin_above = hyperplane_line + margin_offset  # Shift baseline points vertically to mark upper class bounds
margin_below = hyperplane_line - margin_offset  # Shift baseline points vertically to mark lower class bounds
plt.plot(axis_range, margin_above, 'k--', linewidth=1.5, label='Margin Boundaries')  # Plot upper dashed pavement track
plt.plot(axis_range, margin_below, 'k--', linewidth=1.5)  # Plot lower dashed pavement track

# 8. Circle specific points saved in internal model state memory arrays identified as Support Vectors
support_points = svm_model.support_vectors_  # Extract coordinate values saved inside internal model storage fields
plt.scatter(support_points[:, 0], support_points[:, 1], s=250, facecolors='none', edgecolors='black', linewidths=2.5, label='Support Vectors')  # Ring vectors

# 9. Scatter plot original point distributions grouped by target labels over the shaded territory spaces
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', edgecolor='k', s=100, label='Healthy Truck (0)')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', edgecolor='k', s=100, label='Failing Truck (1)')

# 10. Finalize descriptive chart legend titles, labeled axes elements, and screen window layouts
plt.title('Support Vector Machine: Maximum Margin Space Separation')  # Write structural map heading titles
plt.xlabel('Truck Speed (Normalized Scale)')  # Label horizontal feature orientation vectors
plt.ylabel('Vibration Energy (Normalized Scale)')  # Label vertical feature orientation vectors
plt.legend(loc='upper left')  # Draw bounded identifying descriptive map keys
plt.grid(True, linestyle=':', alpha=0.6)  # Layer lightweight structural coordinate alignment dots
plt.show()  # Command standard operating kernels to project finalized map structures onto user screens


