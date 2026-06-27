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


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# 1. Load a sample dataset (Iris)
iris = load_iris()
X, y = iris.data, iris.target

# 2. Create the blueprint for the tree
# We set max_depth to 3 to keep the tree small and readable
clf = DecisionTreeClassifier(max_depth=3)

# 3. Fit the model to the data
clf.fit(X, y)

# 4. Make a prediction for a new, unseen flower
# Features: [sepal length, sepal width, petal length, petal width]
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = clf.predict(new_flower)

print(f"Prediction: {iris.target_names[prediction][0]}")

