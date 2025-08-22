import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense 
from keras.models import Model
 
is_init = False
size = -1

label = []
dictionary = {}
c = 0

# Look for data files in the data directory
# Check if running from src directory or root directory
if os.path.exists("../data"):
    data_dir = "../data"
elif os.path.exists("data"):
    data_dir = "data"
else:
    print("Error: Data directory not found!")
    print("Please ensure you have training data files (.npy) in the data directory.")
    print("Run this script from either the root directory or the src directory.")
    exit(1)

print(f"Using data directory: {data_dir}")

data_files = [f for f in os.listdir(data_dir) if f.endswith('.npy') and f != 'labels.npy']
if not data_files:
	print("Error: No training data files found in the data directory!")
	print("Please collect training data first using data_collection.py")
	exit(1)

print(f"Found {len(data_files)} training data files: {data_files}")

for i in os.listdir(data_dir):
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
		file_path = os.path.join(data_dir, i)
		print(f"Loading data from: {file_path}")
		if not(is_init):
			is_init = True 
			X = np.load(file_path)
			size = X.shape[0]
			y = np.array([i.split('.')[0]]*size).reshape(-1,1)
		else:
			X = np.concatenate((X, np.load(file_path)))
			y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

		label.append(i.split('.')[0])
		dictionary[i.split('.')[0]] = c  
		c = c+1


for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

###  hello = 0 nope = 1 ---> [1,0] ... [0,1]

y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1

print(f"Final data shape: X={X.shape}, y={y.shape}")
print(f"Number of emotions: {len(label)}")
print(f"Emotions: {label}")

ip = Input(shape=(X.shape[1],))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, y, epochs=50)


# Ensure models directory exists
# Check if running from src directory or root directory
if os.path.exists("../models") or data_dir == "../data":
    models_dir = "../models"
else:
    models_dir = "models"

os.makedirs(models_dir, exist_ok=True)
print(f"Using models directory: {models_dir}")

print("Saving model and labels...")
model.save(os.path.join(models_dir, "model.h5"))
np.save(os.path.join(models_dir, "labels.npy"), np.array(label))

print("Training completed successfully!")
print(f"Model saved to: {os.path.join(models_dir, 'model.h5')}")
print(f"Labels saved to: {os.path.join(models_dir, 'labels.npy')}")
print(f"Trained emotions: {label}")