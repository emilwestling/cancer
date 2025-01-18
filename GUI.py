import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

from torchvision import transforms

def preprocess(image):
    transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),  # Resizing
    transforms.ToTensor(),  # Converting to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalizing pixel values
    ])

    return transform(image)
    

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 500)
        self.fc2 = nn.Linear(500, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)  # Flattening step
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
def predict(path):
    image = Image.open(path)
    image = preprocess(image).to(device)

    model = Net()
    state_dict = torch.load('best_model.pt', map_location='mps')
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(image)
        predictions = torch.softmax(predictions, dim=1)
    return torch.argmax(predictions)

from PIL import ImageTk, Image
from tkinter import filedialog

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_path:
        # Display the image
        img = Image.open(file_path).resize((350, 350))  # Resize for GUI display
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Make predictions
        predictions = predict(file_path)
        if predictions == 0:
            prediction_label.config(text="Prediction: Cancer")
        elif predictions == 1:
            prediction_label.config(text="Prediction: Normal")

import tkinter as tk
from tkinter import filedialog

# Initialize the GUI application
root = tk.Tk()
root.title("Lung Cancer CT Scan Detector")
root.geometry("800x600")

# Create GUI components
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

prediction_label = tk.Label(root, text="", font=("Helvetica", 14))
prediction_label.pack(pady=20)

# Run the GUI application
root.mainloop()