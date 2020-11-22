
import cv2
from PIL import Image, ImageTk 
import numpy as np
from keras.preprocessing import image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') 


emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

import tkinter as tk

width, height = 500, 400
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = tk.Tk()
root.geometry("2000x2000")
root.title("facial emotion recognition")
root.configure(background="black")
root.bind('<Escape>', lambda e: root.quit())
l1=tk.Label(root, text="FACIAL EMOTION RECOGNITION",bg="yellow", fg="red",font = "times 30 bold")
l1.pack(side="top", padx=10, pady=10)

l2=tk.Label(root, text="ORIGINAL FACE                                                    EMOJI REACTION",bg="black", fg="blue",font = "times 30 bold")
l2.pack(side="top", padx=10, pady=50)


lmain1 = tk.Label(root)
lmain1.pack(side="left", padx=10)


lmain = tk.Label(root)
lmain.pack(side="right", padx=10)


def show_frame():
    _, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img2 =Image.fromarray(gray)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain1.imgtk2 = imgtk2
    lmain1.configure(image=imgtk2,background="black")
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    #gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGBA)
    #img1 = Image.fromarray(gray1)
    #imgtk = ImageTk.PhotoImage(image=img1)
    #lmain.imgtk = imgtk
    #lmain.configure(image=imgtk,background="black")
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
        #img_pixels =
        #img_pixels = np.array(detected_face)
        #img_pixels=Image.fromarray(Image.fromarray)
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]
        cv2.putText(gray, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        img1=Image.fromarray(gray)
        imgtk = ImageTk.PhotoImage(image=img1)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk,background="black") 
    lmain.after(10, show_frame)
    
		
		

show_frame()
root.mainloop()
		

































import cv2
from PIL import Image, ImageTk 
import numpy as np
from keras.preprocessing import image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') 


emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

import tkinter as tk

width, height = 500, 400
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = tk.Tk()
root.geometry("2000x2000")
root.title("facial emotion recognition")
root.configure(background="black")
root.bind('<Escape>', lambda e: root.quit())
l1=tk.Label(root, text="FACIAL EMOTION RECOGNITION",bg="yellow", fg="red",font = "times 30 bold")
l1.pack(side="top", padx=10, pady=10)

l2=tk.Label(root, text="ORIGINAL FACE                                                    EMOJI REACTION",bg="black", fg="blue",font = "times 30 bold")
l2.pack(side="top", padx=10, pady=50)


lmain1 = tk.Label(root)
lmain1.pack(side="left", padx=10)


lmain = tk.Label(root)
lmain.pack(side="right", padx=10)


def show_frame():
    _, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img2 =Image.fromarray(gray)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain1.imgtk2 = imgtk2
    lmain1.configure(image=imgtk2,background="black")
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    #gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGBA)
    #img1 = Image.fromarray(gray1)
    #imgtk = ImageTk.PhotoImage(image=img1)
    #lmain.imgtk = imgtk
    #lmain.configure(image=imgtk,background="black")
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
        #img_pixels =
        #img_pixels = np.array(detected_face)
        #img_pixels=Image.fromarray(Image.fromarray)
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]
        cv2.putText(gray, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        img1=Image.fromarray(gray)
        imgtk = ImageTk.PhotoImage(image=img1)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk,background="black") 
    lmain.after(10, show_frame)
    
		
		

show_frame()
root.mainloop()
		

