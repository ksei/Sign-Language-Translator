import Tkinter as tk
import cv2
from PIL import Image, ImageTk
import datetime
import imutils
import time
import MultiClassSVM
import utils
import numpy as np


# initialization of the tkinter root window frame
root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
root.title("Sing Language Translator - Test")
root.iconbitmap(r'App_Data/SLTicon.ico')

# initialize open CV camera and frame processing variables
camera = cv2.VideoCapture(0)
camera.set(3, 1280.)
camera.set(4, 720.)
time.sleep(0.25)
frameDelta = None
text = "False"
firstFrame = None
(x1, y1, w1, h1) = (0, 0, 0, 0)

# initialize detection detection required variables
detectionCnt = 0;
path = "Train Data"
charChoice = "A"
lastPrediction = ''
displayText = ""

# method for predicting sign
def predict():
    global frameDelta
    global text
    global charChoice
    global detectionCnt
    global handExited
    global lastPrediction
    global displayText

    # create MultiClass SVM object
    clf = MultiClassSVM.MCSVM()

    # if hand is on the frame proceed with prediction
    if text == "True":
        # get frame
        im = cv2.resize(frameDelta[y1:y1 + h1, x1:x1 + w1], (224, 256))
        #calculate hog for the captured frame
        hog = utils.calc_HOG(im).transpose(1, 0)[0];


        # If character is the same, check if hand has exited and rentered, otherwise continue predicting
        if handExited or (clf.multiClassPredict(hog) != lastPrediction and clf.multiClassPredict(hog) != "O"):
            lastPrediction = clf.multiClassPredict(hog)
            #
            if lastPrediction != "O":
                displayText = displayText + lastPrediction
                output.config(text=displayText)

    handExited = False
    detectionCnt = 3;

# method to clear the displayed text
def clear():
    global displayText

    output.config(text = "")
    displayText = ""

#frame to be passed to Tkinter
def show_frame():
    min_area = 4000;
    global detectionCnt
    global handExited
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

    #get the subframe which contains the gesture
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

    # thresh the image to fill empty spots, then calculate the contours
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

    if detectionCnt > 0:
        detectionCnt = detectionCnt - 1

    # if hand is present predict
    if text == "True" and detectionCnt ==0:
        predict()

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

    # if no hand is detected anymore in the box, set handExited to true
    if text == "False":
        handExited = True


# Tkinter loop
lmain = tk.Label(root)
lmain.pack()

show_frame()

button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, side=tk.BOTTOM)

variable = tk.StringVar(root)
variable.set("A")  # default value

output = tk.Label(button_frame, text="", font=("Helvetica", 20))
output.pack()
output.grid(row=0, column=0)

B = tk.Button(button_frame, text="Clear", command = clear)
B.pack()
B.place(relx=1.0, rely=1.0, anchor='se')


button_frame.pack()

button_frame.focus_set()


root.mainloop()

