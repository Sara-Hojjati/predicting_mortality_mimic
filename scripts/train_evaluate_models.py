#%%
# Import the necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import SGD  # Stochastic Gradient Descent
from tensorflow.keras.optimizers import Adam  # Adam
import numpy as np
import pandas as pd
import shap
from IPython.display import HTML
#%%
#Load the preprocessed data
X_train=pd.read_csv('C:/Users/sarho66/OneDrive - Linköpings universitet/mimic-iii/data/X_train_imputed.csv')
y_train=pd.read_csv('C:/Users/sarho66/OneDrive - Linköpings universitet/mimic-iii/data/y_train.csv')
X_test=pd.read_csv('C:/Users/sarho66/OneDrive - Linköpings universitet/mimic-iii/data/X_test_imputed.csv')
y_test=pd.read_csv('C:/Users/sarho66/OneDrive - Linköpings universitet/mimic-iii/data/y_test.csv')
#%%
# Convert a single-column DataFrame to a Series
y_train = y_train.squeeze()  # If y_train is a DataFrame with one column
y_test = y_test.squeeze()  # If y_test is a DataFrame with one column
#%%
y_train[y_train==1].shape[0]/y_train.shape[0]*100
y_test[y_test==1].shape[0]/y_test.shape[0]*100
print(f" Dead percentage in Train: {y_train[y_train==1].shape[0]/y_train.shape[0]*100:.2f}%, \n Dead percentage in Test: {y_test[y_test==1].shape[0]/y_test.shape[0]*100:.2f}%")
# %%
#Define the model

tf.random.set_seed(123)
model_1 = Sequential([
    # First hidden layer (acts as the input layer with input_shape defined) with 64 neurons
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # Second hidden layer with 32 neurons
    Dense(32, activation='relu'),
    # Output layer with 1 neuron for binary classification
    Dense(1, activation='sigmoid')
])
#%%
model_1 = Sequential([
    # First hidden layer (acts as the input layer with input_shape defined) with 64 neurons
    Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],)),
])
#%%
model_1.summary()
# %%
#Compiling the model


def custom_loss(y_true, y_pred):
    # Define a small epsilon value
    epsilon = 1e-7
    
    # Add epsilon to y_pred to ensure it's never exactly 0 or 1
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    
    # Log the minimum values in y_pred to check for zeros or near-zero values
    tf.print("Min value in y_pred:", tf.reduce_min(y_pred))
    tf.print("Min value in y_true:", tf.reduce_min(y_true))

    # Compute the loss as usual
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Check if the loss becomes NaN
    loss_is_nan = tf.reduce_any(tf.math.is_nan(loss))
    if loss_is_nan:
        tf.print("NaN detected in loss")

    return loss


#%%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 

model_1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#%%
#Training the model
history=model_1.fit(X_train,y_train,epochs=100,verbose=1, validation_split=0.2)

#%%
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Assuming 'history' is the object returned by the 'fit' method of your model
# and contains the training history. Remember, indexing starts at 0,
# so 'after 5 epochs' means starting from index 5 (i.e., epoch 6 onwards).

plt.figure(figsize=(14, 5))

# Plotting training & validation accuracy values starting after epoch 5
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])  # Start from index 5 to skip first 5 epochs
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plotting training & validation loss values starting after epoch 5
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# %%
#Evaluation of the model
test_loss, test_acc= model_1.evaluate(X_test, y_test)
print(f"Test Acuuracy: {test_acc}")
# %%
predictions=model_1.predict(X_test)
# %%[markdown]
# # Interpretation of the model
#%%
X_train_np = X_train[:1000].to_numpy()
X_test_np = X_test[:1000].to_numpy()
# %%
# Initialize the SHAP explainer
explainer = shap.DeepExplainer(model_1, X_train_np[:1000])
#%%
# Calculate SHAP values
shap_values = explainer.shap_values(X_test_np[:1000])


# %%
# Plot the SHAP values
shap.summary_plot(shap_values, X_test[:1000])
# %%
#%%
# Summary plot for each class
for i in range(len(shap_values)):  # Assuming shap_values is a list where each item corresponds to a class
    print(f"Class {i} Summary Plot")
    shap.summary_plot(shap_values[i], X_test[:1000])
# %%
for i in range(X_test_np.shape[1]):  # Loop through each feature
    shap.dependence_plot(i, shap_values[0], X_test[:1000])

# %%
# For a single prediction

# %%
# Make plot for a single prediction
# Initialize SHAP's JavaScript visualization
shap.initjs()

# Define the index of the sample you're interested in
sample_index = 1  # Adjust this to your specific sample

# Assuming explainer.expected_value is an EagerTensor or similar
expected_value_serializable = explainer.expected_value[0].numpy() if hasattr(explainer.expected_value[0], 'numpy') else explainer.expected_value[0]

# If shap_values[0][0] is an EagerTensor
shap_values_serializable = shap_values[0][sample_index].numpy() if hasattr(shap_values[0][sample_index], 'numpy') else shap_values[0][sample_index]

# Keeping X_test as a DataFrame row to preserve feature names
X_test_serializable = X_test.iloc[sample_index] if isinstance(X_test, pd.DataFrame) else X_test[sample_index]

# Convert X_test_serializable to numpy if needed, and if it's a pandas Series, ensure it remains a Series to preserve feature names
if hasattr(X_test_serializable, 'numpy'):
    X_test_serializable = X_test_serializable.to_numpy()
elif isinstance(X_test_serializable, np.ndarray):
    # Convert to a list or keep as numpy array; shap.force_plot can handle both
    pass
else:
    # Ensure X_test_serializable is in a suitable format for shap.force_plot
    X_test_serializable = X_test_serializable.values

# Generate the SHAP force plot with feature names
force_plot = shap.force_plot(expected_value_serializable, shap_values_serializable, X_test_serializable, feature_names=X_train.columns.tolist())
force_plot
# %%
# Save the SHAP force plot as an HTML file
shap.save_html('C:/Users/sarho66/OneDrive - Linköpings universitet/mimic-iii/results/force_plot_1.html', force_plot)
# %%