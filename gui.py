from pathlib import Path
from tkinter import Tk, Canvas, Button, filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import pickle
import cv2
import numpy as np

sz = 28
hiragana = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        pil_image = Image.open(file_path)
        pil_image_resized = pil_image.resize((250, 250))
        new_image = ImageTk.PhotoImage(pil_image_resized)
        canvas.itemconfig(image_2, image=new_image)
        canvas.image2 = new_image
        
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        preprocessing_image = data_preprocessing(opencv_image)
        
        preprocessing_image_pil = Image.fromarray(preprocessing_image)
        preprocessing_image_pil_resized = preprocessing_image_pil.resize((250, 250))
        preprocessing_image_tk = ImageTk.PhotoImage(preprocessing_image_pil_resized)
        canvas.itemconfig(image_3, image=preprocessing_image_tk)
        canvas.image3 = preprocessing_image_tk

        predict(preprocessing_image)

def data_preprocessing(img, target_size=(sz, sz)):
    _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(img_binary)
    x, y, w, h = cv2.boundingRect(coords)
    img_crop = img_binary[y:y+h, x:x+w]
    resized_img = cv2.resize(img_crop, target_size, interpolation=cv2.INTER_AREA)
    return resized_img

def predict(img):
    y_pred = -1
    if current_model != cnn_model:
        x_pred = np.array([np.array(img)])
        x_pred = x_pred.reshape(x_pred.shape[0], -1)
        y_pred = current_model.predict(x_pred)
    else:
        x_pred = img_to_array(img)
        x_pred = x_pred / 255.0  # 進行 rescale
        x_pred = np.expand_dims(x_pred, axis=0)  # 增加一個維度以符合模型的輸入要求

        pred = current_model.predict(x_pred)
        print(f'Predict: {pred[0]}')
        y_pred = np.argmax(pred, axis=1)

    canvas.itemconfig(predict_text, text=f'Predict: {y_pred[0]}, {hiragana[y_pred[0]]}')
    return

def update_model(model, text):
    global current_model
    current_model = model
    canvas.itemconfig(current_model_text, text=text)

svm_model = None
with open(f'../models/svm_model_0.8678.pkl', 'rb') as file:
    svm_model = pickle.load(file)

rf_model = None
with open(f'../models/rf_model_0.8368.pkl', 'rb') as file:
    rf_model = pickle.load(file)

cnn_model = load_model('../models/cnn_model_9814_with_detect.h5')

current_model = svm_model

window = Tk()
window.geometry("696x695")
window.configure(bg = "#FFFFFF")

canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 695,
    width = 696,
    bd = 0,
    highlightthickness=0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)

image_image_1 = ImageTk.PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    348.0,
    347.0,
    image=image_image_1
)

image_image_2 = ImageTk.PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    190.0,
    275.0,
    image=image_image_2
)

image_image_3 = ImageTk.PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    505.0,
    275.0,
    image=image_image_3
)

button_image_1 = ImageTk.PhotoImage(
    file=relative_to_assets("svm.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: update_model(svm_model, "Current Model: SVM"),
    relief="flat"
)
button_1.place(
    x=52.0,
    y=455.0,
    width=179.0,
    height=36.0
)

button_image_2 = ImageTk.PhotoImage(
    file=relative_to_assets("rf.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: update_model(rf_model, "Current Model: RF"),
    relief="flat"
)
button_2.place(
    x=258.0,
    y=455.0,
    width=179.0,
    height=36.0
)

button_image_3 = ImageTk.PhotoImage(
    file=relative_to_assets("cnn.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: update_model(cnn_model, "Current Model: CNN"),
    relief="flat"
)
button_3.place(
    x=464.0,
    y=456.0,
    width=179.0,
    height=36.0
)

button_image_4 = ImageTk.PhotoImage(
    file=relative_to_assets("upload.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=upload_image,
    relief="flat"
)
button_4.place(
    x=230.0,
    y=525.0,
    width=235.8253936767578,
    height=36.0
)

current_model_text = canvas.create_text(
    118.0,
    22.0,
    anchor="nw",
    text="Current Model: SVM",
    fill="#005384",
    font=("Quantico", 48 * -1)
)

predict_text = canvas.create_text(
    190.0,
    595.0,
    anchor="nw",
    text="Predict:",
    fill="#005384",
    font=("Quantico", 48 * -1)
)

window.resizable(False, False)
window.mainloop()
