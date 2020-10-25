from tkinter import *
from tkinter import ttk
from tkinter import messagebox

root=Tk()
def Kfunc():
    mbox1=messagebox.askquestion('Run','Run K-Nearest Neighbour?')
    if mbox1 == 'yes':
        import knn

def Nfunc():
    mbox2 = messagebox.askquestion('Run', 'Run Naive Bayes?')
    if mbox2 == 'yes':
       import naive

def Lfunc():
    mbox3 = messagebox.askquestion('Run', 'Run Logistic Regression?')
    if mbox3 == 'yes':
        import logistic

def FuncExit():
    mbox = messagebox.askquestion('Exit', 'Are you to exit?')
    if mbox == 'yes':
        root.destroy()


root.title('Prediction')
title=Label(root,text="Prediction of Robot's Action",font='verdana 16 bold underline')
label=Label(root,text='Select Algorithm :')
buttonk=ttk.Button(root,text='K-Nearest Neighbour',command=Kfunc)
buttonn=ttk.Button(root,text='Naive Bayes',command=Nfunc)
buttonl=ttk.Button(root,text='Logistic Regression',command=Lfunc)
buttonk.place(x=150,y=140)
buttonn.place(x=150,y=180)
buttonl.place(x=150,y=220)
button1=ttk.Button(root,text='Exit',command=FuncExit)
label.place(x=20,y=140)
title.place(x=80,y=20)
button1.place(x=180,y=350)

root.geometry("450x450+650+350")
root.mainloop()