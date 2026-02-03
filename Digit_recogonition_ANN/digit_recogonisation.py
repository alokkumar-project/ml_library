from tkinter import *
import pandas as pd
from tkinter import messagebox as msg
from tkinter import filedialog
from alok_digit import *
from alok_digit import convert_image as ci
class Ai:
    def train(self):
        p.config(state=DISABLED)
        dataset = pd.read_csv('mnist_train.csv')
        x_train = dataset.iloc[:,1:]
        y_train = dataset.iloc[:,0]

        x_train = x_train/255
        x_train = x_train.values.reshape(-1,784)
        self.n = Nueral_Network(x_train,y_train,nueron=[128,64,32,10],opti='Adam',epoch=1500)
        msg.showinfo("Training", "Trained_successfully.")
    def upload_file(self):
        self.file_path = filedialog.askopenfilename(title="Select your image",filetypes=[("Image files","*.jpeg")])
        return self.file_path
    def predict(self):
        #print(self.file_path)
        value = ci(self.file_path)
        output = self.n.predict(value)
        msg.showinfo("Prediction",output)

    
root = Tk()
root.title("Digit Recogonisation")
root.geometry("400x200")
root.minsize(400,200)
root.maxsize(500,200)
icon_image = PhotoImage(file="rob.png")
root.iconphoto(True,icon_image)
a = Label(text="Upload your image")
a.pack()
maya = Ai()
p = Button(fg='green',text="Train model", command=maya.train)
p.pack()
b = Button(fg="red",text="Select Image",command=maya.upload_file)
b.pack()
pre = Button(fg="blue",text="Predict Number",command=maya.predict)
pre.pack()
root.mainloop()