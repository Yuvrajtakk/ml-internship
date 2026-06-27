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


import matplotlib.pyplot as plt

inputs = [1, 2, 3, 4]
targets = [5, 7, 9, 11]

# The final parameters our machine learned
w = 2.0
b = 3.0

# 1. List Comprehension: A fast, Pythonic way to run a loop inside a list
# This calculates the predicted y value for every x in our inputs
predictions = [(x * w) + b for x in inputs]

# 2. Scatter plot: Draws our original dataset as individual dots
plt.scatter(inputs, targets, color='blue', label='Actual Data')

# 3. Line plot: Draws our machine's learned rule as a continuous line
plt.plot(inputs, predictions, color='red', label='Learned Rule')

# 4. Make the graph readable
plt.xlabel('Inputs (x)')
plt.ylabel('Targets (y)')
plt.title('Machine Learning: Fitting a Line to Data')
plt.legend()

# 5. Render the window on your screen
plt.show()










