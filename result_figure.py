import matplotlib.pyplot as plt

# Preparing the data for plotting
accuracy_values = [92.95, 95.95, 96.55, 96.75, 96.45, 95.70, 95.25, 95.25, 95.00, 95.35]
# Converting accuracy percentages to decimals
accuracy_values = [x / 100 for x in accuracy_values]
precision_values = [0.67, 0.80, 0.82, 0.83, 0.82, 0.78, 0.76, 0.76, 0.74, 0.76]
recall_values = [0.96, 0.94, 0.96, 0.95, 0.95, 0.96, 0.96, 0.96, 0.97, 0.97]
f1_values = [0.79, 0.86, 0.88, 0.89, 0.88, 0.86, 0.85, 0.85, 0.84, 0.85]
# Epochs from 1 to 10
epochs = list(range(1, 11))

# Loss values for each epoch
loss_values = [0.55466, 0.14065, 0.07822, 0.04447, 0.0248, 0.01369, 0.00892, 0.00702, 0.00569, 0.00394]

# Creating the plot
plt.figure(figsize=(12, 8))
plt.plot(epochs, loss_values, marker='o', color='b', linestyle='-', label='Loss')
plt.plot(epochs, accuracy_values, marker='s', color='g', linestyle='-', label='Accuracy')
plt.plot(epochs, precision_values, marker='^', color='r', linestyle='-', label='Precision')
plt.plot(epochs, recall_values, marker='v', color='m', linestyle='-', label='Recall')
plt.plot(epochs, f1_values, marker='d', color='c', linestyle='-', label='F1 Score')

# Adding titles and labels
plt.title('Metrics per Epoch')
plt.xlabel('Epoch')
# plt.ylabel('Values')  # Omitting y-axis label as requested

# Legend
plt.legend()

# Removing y-axis ticks as requested
plt.yticks([])

# Saving the plot to a file
plt.savefig('./metrics_per_epoch.png', format='png')

# Showing the plot
plt.show()
