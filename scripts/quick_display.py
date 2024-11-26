# -*- coding: utf-8 -*-
"""This script was created for quickly looking through nifti files given a filename and path location, all three\
    planes can be viewed and the slices can be scrolled through, the data is set automatically to be time averaged,
    but individual timepoints can also be dispalayed"""
# IMPORTS
import os
import nibabel as nib
from matplotlib.figure import Figure
import numpy as np
import matplotlib, sys
from matplotlib.backends._backend_tk import NavigationToolbar2Tk

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
# ----------------------------------------------------------------------------
# Placeholder variables
pname = r'C:\Users\Kaitlin\Desktop\python_training'
fname = 'Series13.nii'


# ----------------------------------------------------------------------------
# initiate main gui window with one frame (MainDisplay)

class QDISP(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default='lablogoicon.ico')
        tk.Tk.wm_title(self, "Quick Display Nifti1")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        frame = MainDisplay(container, self)

        self.frames[MainDisplay] = frame

        frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(MainDisplay)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


# Main frame class, contains initialization, canvas, and configure canvas controls

class MainDisplay(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # name the main frame and display label in top left
        label = tk.Label(self, text="Main Window")
        label.grid(row=0, column=0, pady=10, padx=10)
        # configure main frame for resizing and set a minimum size of 10x10
        for row_num in range(self.grid_size()[1]):
            self.rowconfigure(row_num, minsize=10, weight=2)
        for col_num in range(self.grid_size()[0]):
            self.columnconfigure(col_num, minsize=7, weight=2)
        # class attributes for identification of selected option
        self.plane_var = tk.IntVar()
        self.slice_var = tk.IntVar()
        self.time_var = tk.IntVar()
        self.timepoint_var = tk.IntVar()
        self.input_data, self.taveraged_data, self.dims = self.prep_data
        # run functions with initial parameters
        # setup frames
        self.options_frame, self.box_frame_plane, self.box_frame_time = self.create_frames()
        # setup buttons and commands
        self.max_slice = self.scale_max()
        self.quit_button, self.plane_options_buttons, self.time_option_buttons, self.slice_scale, \
        self.time_point_sel_box = self.create_selection_widgets()
        self.set_callbacks()
        print(self.dims[0])
        # disp canvas
        self.make_img_canvas()

    def create_frames(self):
        """ Create all the frames and sub-frames within the main window and configure for resizing"""
        # Canvas Options frame, holds all selection buttons
        options_frame = tk.Frame(self)
        options_frame.grid(row=1, column=2)
        # Frame for plane selection radio buttons
        box_frame_plane = tk.LabelFrame(options_frame, text="Plane Selection")
        box_frame_plane.grid(row=0, column=0, sticky="NSEW")
        # Frame for individual / average radio buttons
        box_frame_time = tk.LabelFrame(options_frame, text="Time Selection")
        box_frame_time.grid(row=1, column=0, sticky="NSEW")
        return options_frame, box_frame_plane, box_frame_time

    def create_selection_widgets(self):
        """make, and display all the buttons and sliders
        :returns quit_button, plane_option_buttons, time_option_buttons, slice_scale, time_point_sel_box:"""
        # Plane selection Radiobutton
        planes = [("Sagital", 1), ("Axial", 2), ("Coronal", 3)]
        plane_option_buttons = []
        for p_val, plane in enumerate(planes):
            plane_button = tk.Radiobutton(self.box_frame_plane, text=plane[0], value=p_val)
            plane_button.grid(row=0, column=p_val)
            plane_option_buttons.append(plane_button)

        # Averaged or Individual Time Radiobutton
        temporal_opts = [("Time Averaged", 1), ("Individual Timepoints", 2)]
        time_option_buttons = []
        for t_val, t_opt in enumerate(temporal_opts):
            time_button = tk.Radiobutton(self.box_frame_time, text=t_opt[0], value=t_val)
            time_button.grid(row=0, column=t_val)
            time_option_buttons.append(time_button)
        # Window action buttons
        quit_button = tk.Button(self, width=15, text="Quit")
        quit_button.grid(row=3, column=3, sticky="NSEW")
        # scale bar for scrolling through slices
        slice_scale = tk.Scale(self, from_=1, to=self.dims[0], variable=self.slice_var, command=self.slice_select)
        slice_scale.grid(row=1, column=1, sticky="NSW")
        # time point selection options
        time_point_sel_box = tk.Spinbox(self.box_frame_time, from_=1, to=self.dims[3])
        time_point_sel_box.grid(row=0, column=3)
        time_point_sel_button = tk.Button(self.box_frame_time, text="Select Time", command=self.make_img_canvas)
        time_point_sel_button.grid(row=0, column=4)
        return quit_button, plane_option_buttons, time_option_buttons, slice_scale, time_point_sel_box

    def set_callbacks(self):
        """set the commands associated with each button"""
        for plane_button in self.plane_options_buttons:
            plane_button['variable'] = self.plane_var
            plane_button['command'] = lambda x=enumerate(self.plane_options_buttons): self.make_img_canvas(x)
        for time_button in self.time_option_buttons:
            time_button['variable'] = self.time_var
            time_button['command'] = lambda y=enumerate(self.time_option_buttons): self.temporal_select(y)
        self.slice_scale.bind("<ButtonRelease-1>", lambda z=self.slice_var: self.make_img_canvas(z))
        spin_val = int(self.time_point_sel_box.get())
        print(spin_val)
        self.quit_button['command'] = self.quit

    def scale_max(self):
        """:return max_slice (int) """
        x = self.plane_var.get()
        max_slice = self.dims[x]
        return max_slice

    @property
    def prep_data(self):
        """ Prepare the data by importing it formating as an array, creating averaged set, and get dimensions
        :return input_data: array formed from Nifti img data
                taveraged_data: input_data averaged over time
                dims: dimensions of input image ( sagital,coronal,axial,time)
        """
        # get filename, prepare data for display
        fullname = os.path.join(pname, fname)
        print('Input filename is: {name}'.format(name=fullname))
        input_img = nib.load(fullname)
        input_data = input_img.get_fdata()
        input_hdr = input_img.header
        np.shape(input_data)
        dims = input_hdr.get_data_shape()
        taveraged_data = np.average(input_data, axis=3)
        return input_data, taveraged_data, dims

    def slice_select(self, *args, **kwargs):
        """ Selects slice of volume that was scrolled to
            :return: slice_selection (int) : the number to change in the imshow(data_selection) ,
                                            varies with plane selected """
        z = self.slice_var.get()
        slice_selection = z
        return slice_selection

    def temporal_select(self, *args, **kwargs):
        """ Selects averaged or unaveraged data and if unaveraged, selects time point
            :returns:
            selected_t: image data array
            x : value of  0 or 1 ,where  0=avg
            t = value from spinbox -1 , indicates volume number"""
        x = self.time_var.get()
        t = tk.IntVar()
        if x == 0:
            print("t_value is avg")
            selected_t = self.taveraged_data
            self.time_point_sel_box['state'] = tk.DISABLED
            t = None
        elif x == 1:
            print("t_value is ind")
            selected_t = self.input_data
            self.time_point_sel_box['state'] = tk.NORMAL
            t = int(self.time_point_sel_box.get()) - 1
            print(t)
        else:
            print("temp_select no t_val selected")
        return selected_t, x, t

    def make_img_canvas(self, *args, **kwargs):
        """Plots the image to the canvas according to the selected options for plane, time and slice"""
        selected_t, x, t = self.temporal_select()
        selected_plane = self.plane_var.get()
        selected_slice = self.slice_select()
        fig = Figure(figsize=(5, 5))
        fig.clear()
        a = fig.add_subplot(111)
        # plane select is done in here to save a def... imshow is fussy

        if selected_plane == 0:
            max_slice = self.dims[0]
            self.slice_scale.configure(to=max_slice)
            if x == 0:
                volume_selected = selected_t[selected_slice, :, :]
            elif x == 1:
                print(selected_slice)
                print(t)
                volume_selected = selected_t[selected_slice, :, :, int(t)]
            else:
                print("invalid temporal selection var 'x'")

        elif selected_plane == 1:
            max_slice = self.dims[2]
            self.slice_scale.configure(to=max_slice)
            if x == 0:
                volume_selected = selected_t[:, :, selected_slice]
            elif x == 1:
                print(selected_slice)
                print(t)
                volume_selected = selected_t[:, :, selected_slice, int(t)]
            else:
                print("invalid temporal selection var 'x'")
        elif selected_plane == 2:
            max_slice = self.dims[1]
            self.slice_scale.configure(to=max_slice)
            if x == 0:
                volume_selected = selected_t[:, selected_slice, :]
            elif x == 1:
                print(selected_slice)
                print(t)
                volume_selected = selected_t[:, selected_slice, :, int(t)]
            else:
                print("invalid temporal selection var 'x'")

        else:
            print("Error invalid plane_selection")
        a.imshow(volume_selected, 'gray')
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().grid(row=1, column=0, sticky="NSEW")
        canvas.draw()


#     # TODO: make figure creation button
#     # TODO: add textbox with pwd and filename


if __name__ == "__main__":
    app = QDISP()
    app.mainloop()
