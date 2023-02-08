# testing how to make GUI widgets visible on all platforms

import tkinter as tk
import os
import re
import matplotlib

matplotlib.use('TkAgg')   # explicitly set this - it might help with displaying figures in different environments

# save some colors for consistent layout, and make them easy to change
# colours for Windows
if os.name == 'nt':
    fgcol1 = 'navy'
    fgcol2 = 'gold2'
    fgcol3 = 'firebrick4'
    bgcol = 'grey94'
    fgletter1 = 'white'
    fgletter2 = 'black'
    fgletter3 = 'white'
    widgetfont = "none 9 bold"
    widgetfont2 = "none 9 bold"
    labelfont = "none 9 bold"
    radiofont = "none 9"
    infofont = "none 8"
    widgetbg = 'grey94'
else:
    # colours for Mac/Linux
    fgcol1 = 'red'
    fgcol2 = 'red'
    fgcol3 = 'red'
    bgcol = 'grey94'
    fgletter1 = 'black'
    fgletter2 = 'black'
    fgletter3 = 'black'
    widgetfont = "none 9 bold"
    widgetfont2 = "none 9 bold"
    labelfont = "none 9 bold"
    radiofont = "none 9"
    infofont = "none 9"
    widgetbg = '#3E4149'
    widgetbg = '#6a0000'

bigbigbuttonsize = 21
bigbuttonsize = 14
smallbuttonsize = 9

# define a single place for saving setup parameters for ease of retrieving, updating, etc.
basedir = os.getcwd()

# ------Create the Base Window that will hold everything, widgets, etc.---------------------
class main_window:
    # defines the main window, and other windows for functions are defined in separate classes

    def __init__(self, parent):
        parent.configure(relief='raised', bd = 5, highlightcolor = fgcol3)
        self.parent = parent

        # initialize some values
        # put some text as a place-holder
        self.label1 = tk.Label(self.parent, text = "Testing text", fg = 'gray')
        self.label1.grid(row=0,column=0, sticky='W')

        # create the Entry box, and put it next to the label, 4th row, 2nd column
        self.enter1 = tk.Entry(self.parent, width = 20, bg="white")
        self.enter1.grid(row=0, column = 2, sticky = "W")
        self.enter1.insert(0,'testing entry box')

        # the entry box needs a "submit" button so that the program knows when to take the entered values
        self.enter1submit = tk.Button(self.parent, text = "Submit", width = smallbuttonsize, bg = fgcol2, fg = fgletter2, command = self.enter1submitclick, relief='raised', bd = 5, highlightbackground = widgetbg)
        self.enter1submit.grid(row = 0, column = 3)


        # testing radio buttons
        # radioboxes to indicate stats correction choices
        self.label2 = tk.Label(self.parent, text="Testing check boxes:", font = labelfont)
        self.label2.grid(row=2, column=1, sticky='W')
        self.radio1value = tk.IntVar(None,1)
        self.radio1 = tk.Radiobutton(self.parent, text = 'choice 1', width = smallbuttonsize, bg = bgcol, fg = fgletter2, font = radiofont,
                                          command = self.setradiovalue, variable = self.radio1value, value = 1, state = tk.NORMAL, highlightbackground = widgetbg)
        self.radio1.grid(row = 2, column = 2, sticky="E")
        self.radio2 = tk.Radiobutton(self.parent, text = 'choice 2', width = smallbuttonsize, bg = bgcol, fg = fgletter2, font = radiofont,
                                          command = self.setradiovalue, variable = self.radio1value, value = 2, state = tk.NORMAL, highlightbackground = widgetbg)
        self.radio2.grid(row = 2, column = 3, sticky="E")
        self.radio3 = tk.Radiobutton(self.parent, text = 'choice 3', width = smallbuttonsize, bg = bgcol, fg = fgletter2, font = radiofont,
                                          command = self.setradiovalue, variable = self.radio1value, value = 3, state = tk.NORMAL, highlightbackground = widgetbg)
        self.radio3.grid(row = 2, column = 4, sticky="E")

        # pull-down menu
        self.fieldvalues = ['choice 1', 'choice 2', 'choice X', 'no choice']
        self.pulldown1_var = tk.StringVar()
        self.pulldown1_var.set(self.fieldvalues[0])
        pulldown1_menu = tk.OptionMenu(self.parent, self.pulldown1_var, *self.fieldvalues,
                                        command=self.pulldown1valuechoice)
        pulldown1_menu.grid(row=3, column=1, sticky='EW')
        self.pulldown1_var_opt = pulldown1_menu  # save this way so that values are not cleared

    # action when the button is pressed to submit the DB entry number list
    def enter1submitclick(self):
        # first load the settings file so that values can be used later
        entered_text = self.enter1.get()  # collect the text from the text entry box
        # remove any spaces
        entered_text = re.sub('\ +','',entered_text)
        print(entered_text)

        # update the text in the box, in case it has changed
        self.enter1.delete(0,'end')
        self.enter1.insert(0,entered_text)

        if entered_text[:4].lower() == 'hide':
            self.radio1.config(state=tk.DISABLED)
            self.radio2.config(state=tk.DISABLED)
            self.radio3.config(state=tk.DISABLED)
        else:
            self.radio1.config(state=tk.NORMAL)
            self.radio2.config(state=tk.NORMAL)
            self.radio3.config(state=tk.NORMAL)


    def setradiovalue(self):
        value = self.radio1value.get()
        if value == 1:
            self.radio1choice = 'choice1'

        if value == 2:
            self.radio1choice = 'choice2'

        if value == 3:
            self.radio1choice = 'choice3'

        print('choice set to {} which is {}'.format(value,self.radio1choice))
        return self


    def pulldown1valuechoice(self,value):
        # get the field value choices for the selected field
        self.pulldown1value = self.pulldown1_var.get()
        print('Selected option: {}'.format(self.pulldown1value))

        # # destroy the old pulldown menu and create a new one with the new choices
        # rownum = 3
        # columnum = 1
        # self.pulldown1_var = tk.StringVar()
        # self.pulldown1_var.set(self.fieldvalues[0])
        # self.pulldown1_var_opt.destroy()  # remove it
        # pulldown1_menu = tk.OptionMenu(self.parent, self.pulldown1_var, *self.fieldvalues,
        #                                 command=self.pulldown1valuechoice)
        # pulldown1_menu.grid(row=rownum, column=columnum, sticky='EW')
        # self.pulldown1_var_opt = pulldown1_menu  # save this way so that values are not cleared
        return self



# ----------MAIN calling function----------------------------------------------------
# the main function that starts everything running
def main():
    root = tk.Tk()
    root.title('Testing Visibility')
    app = main_window(root)
    root.mainloop()

if __name__ == '__main__':
    main()

