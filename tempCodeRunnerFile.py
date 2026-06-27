
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