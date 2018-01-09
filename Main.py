import sys
import os
import Tkinter as tk
import tkMessageBox
import subprocess
from PIL import Image, ImageTk



def trainCallBack():
    App.root.destroy()
    proc = subprocess.Popen([sys.executable, 'Train.py'],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)

    while True:
        out = proc.stdout.read(1)
        if out == '' and proc.poll() != None:
            break
        if out != '':
            sys.stdout.write(out)
            sys.stdout.flush()

    #os.system('python Train.py')


def testCallBack():
    App.root.destroy()
    subprocess.Popen([sys.executable, 'Tester.py'],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)



class ASLMain:
    def __init__(self):

        self.root = tk.Tk()
        self.train_icon = Image.open('App_Data/TrainBtn.png')
        self.train_im = ImageTk.PhotoImage(self.train_icon)
        self.test_icon = Image.open('App_Data/TranslateBtn.png')
        self.test_im = ImageTk.PhotoImage(self.test_icon)
        self.root.title("ASL Translator - Main")
        self.root.iconbitmap(r'App_Data/SLTicon.ico')
        self.root.geometry("600x354")
        self.root.update()
        self.trainBtn = tk.Button(self.root,command= trainCallBack, width=1, height=1)
        self.trainBtn.config(image=self.train_im)
        self.trainBtn.pack(fill=tk.BOTH,side=tk.LEFT, expand=True)
        self.testBtn = tk.Button(self.root,command= testCallBack, width=1, height=1)
        self.testBtn.config(image=self.test_im)
        self.testBtn.pack(fill=tk.BOTH,side=tk.RIGHT, expand=True)
        # check if training has been completed
        if not os.path.exists('storedSVM/model.npz'):
            self.testBtn.config(state="disabled");

App = ASLMain()
App.root.mainloop()