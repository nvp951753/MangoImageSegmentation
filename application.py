from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from tkinter import filedialog
import nhandiendef as fil
import numpy as np
import tkinter as tk
def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports
win = Tk()
win.geometry("900x400")
win.title("Nhận diện phần khuyết tật")
avai_cam = list_ports()
avai_cam = avai_cam[1]
items = []
for i in avai_cam:
    items.append("cam"+str(i))
com3 = ttk.Combobox(win,values=items)
com3.current(0)
com3.place(x=5, y=10, width=80)

label = Label(win)

def to_pil(img,label,x,y,w,h):
    img = cv2.resize(img, (w, h))
    image = Image.fromarray(img)
    pic = ImageTk.PhotoImage(image)
    label.configure(image=pic)
    label.image = pic
    label.place(x=x, y=y)

def choose():
    global cap
    for device in items:
        if com3.get()==device:
            cap =cv2.VideoCapture(int(device[3:]))
            break
    show()
defe = tk.StringVar()
defe.set("Defect: %")
button3 =Button(win,text='switch',command=choose)
button3.place(x=100,y=10)
text = Label(win, text="Original \n Image")
text.place(x=40,y=65+128)
text2 = Label(win, text="Mask Image")
text2.place(x=40+128,y=65+128)
text3 = Label(win, text="Masked Image")
text3.place(x=40+256,y=65+128)
text4 = Label(win, text="Defect's Mask")
text4.place(x=40+256+128,y=65+128)
text5 = Label(win, text="Masked Defect")
text5.place(x=40+512,y=65+128)
text6 = Label(win, textvariable=defe)
text6.place(x=40,y=65+168)
imgs = "mask"
switch = ttk.Combobox(win,values=imgs)
switch.place(x=400, y=10, width=80)
switch.current(0)

def show():
    _, frame = cap.read()

    global dientich
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    final, dientich = fil.badfilter(rgb)
    defe.set("Defect: "+str(round(dientich*100,2))+ "%")

    x,y= final.shape[:2]
    to_pil(final,label,10,50,y,x)
    label.after(30,show)


win.mainloop()
