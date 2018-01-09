import Tkinter as tk
import cv2
from PIL import Image, ImageTk
import datetime
import imutils
import time
import MultiClassSVM
import subprocess
import sys
import utils

root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
root.title("Sing Language Translator - Train")
root.iconbitmap(r'App_Data/SLTicon.ico')

camera = cv2.VideoCapture(0)
camera.set(3, 1280.)
camera.set(4, 720.)
time.sleep(0.25)
path = "Train Data"
frameDelta = None;
text = "False"
firstFrame = None;
charChoice = "A";
(x1, y1, w1, h1) = (0, 0, 0, 0)


def storeSample(event):
    global frameDelta
    global text
    global charChoice

    current = utils.getGestureNo(charChoice)


    if (current == ''):
        current = 1
    else:
        current = int(current) + 1
    if text == "True":
        cv2.imwrite(path + '\\' + charChoice + str(current) + ".jpg",
                    cv2.resize(frameDelta[y1:y1 + h1, x1:x1 + w1], (224, 256)))
        l1.config(text = "Sample " + charChoice + str(current) + " stored ...")


def startTraining():

    w.destroy()
    B.destroy()
    l.config(text='Please wait while SVM is creating the new model...')
    root.destroy()
    msvm = MultiClassSVM.MCSVM();
    msvm.multiClassFit()



    subprocess.Popen([sys.executable, 'Tester.py'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
    quit()


def change_dropdown(*args):
    global charChoice
    charChoice = variable.get()
    displayText = charChoice
    l1.config(text=displayText + " selected")



def show_frame():
    min_area = 500;

    global text
    global frameDelta
    global charChoice
    global firstFrame
    global x1, y1, w1, h1
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        return
    text = "False"
    frame = cv2.flip(frame, 1);
    subframe = frame[160:640, 790:1110]
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=1280)
    gray = cv2.cvtColor(subframe, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the threshold image to fill in holes, then find contours
    # on threshold image
    thresh = cv2.dilate(thresh, None, iterations=2)
    ( _, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) > min_area:
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            (x1, y1, w1, h1) = (x, y, w, h)

            (x, y, w, h) = ((x + 790) * 1280 / 1280, (y + 160) * 1280 / 1280, w * 1280 / 1280, h * 1280 / 1280)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            text = "True"


    # draw the text and timestamp on the frame
    cv2.putText(frame, "Hand Presence: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.rectangle(frame, (790 * 1280 / 1280, 160 * 1280 / 1280), (1110 * 1280 / 1280, 640 * 1280 / 1280),
                  (0, 255, 0), 3)

    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
    # cv2.imshow("Subframe", subframe)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

# start Tkinter Loop

lmain = tk.Label(root)
lmain.pack()

show_frame()

button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, side=tk.BOTTOM)

variable = tk.StringVar(root)
variable.set("A")  # default value

l = tk.Label(button_frame, text="Choose a letter:")
l.pack()
l.grid(row=0, column=0)

l1 = tk.Label(button_frame, text="No Letter Selected")
l1.pack()
l1.grid(row=0)
l1.place(relx=1.0, rely=1.0, anchor='se')

w = tk.OptionMenu(button_frame, variable, "A", "B", "C", "D", "E", "F", "G", "H", "I", "O")
w.pack()

B = tk.Button(button_frame, text="Complete Sampling", command = startTraining)
B.pack()

w.columnconfigure(0, weight=1)
B.columnconfigure(1, weight=1)

w.grid(row=0, column=1, sticky=tk.W + tk.E)
B.grid(row=0, column=2, sticky=tk.W + tk.E)

variable.trace('w', change_dropdown)

# bind 'C' to handler
button_frame.bind("<KeyPress-c>", storeSample)
button_frame.pack()

button_frame.focus_set()


root.mainloop()

