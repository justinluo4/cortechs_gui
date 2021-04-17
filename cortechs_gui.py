import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.artist
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import math
import tkinter as tk

# Read all used data, use local file path instead
ALL_DATA = pd.read_csv(r"C:\Users\justi\Downloads\hrrt_fdg_check2_FDG_HRRT_PETQuant_combined (1).csv")
Z_SCORE_TABLE = pd.read_csv(
    r"C:\Users\justi\Downloads\Untitled spreadsheet - B211 Z Scores & Normal Distribution (2).csv")
NORMAL_DATA = pd.read_csv(r"C:\Users\justi\Downloads\Normal_Model.csv")
TEST_DATA = pd.read_csv(r"C:\Users\justi\Downloads\test_data.csv")
ALL_DATA = ALL_DATA.sort_values(by=['Patient ID', 'AGE'])
ALL_DATA = ALL_DATA.reset_index(drop=True)


class Visualizer(object):
    def __init__(self):
        # Create window & matplotlib canvas
        fig = plt.figure()
        self.fig = fig
        root = tk.Tk()
        root.title("Depth Calculator")
        self.canvas = FigureCanvasTkAgg(fig, root)
        self.canvas.get_tk_widget().pack(side="top", fill='both', expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        # List used brain structures
        self.all_labels = ["Cerebral White Matter", "Cortical Gray Matter",
                           "3rd Ventricle", "4th Ventricle",
                           "5th Ventricle", "Superior Lateral Ventricle",
                           "Inferior Lateral Ventricle", "SLV Choroid Plexus",
                           "Cerebellar White Matter", "Cerebellar Gray Matter",
                           "Hippocampus", "Amygdala",
                           "Thalamus", "Caudate",
                           "Putamen", "Pallidum",
                           "Ventral Diencephalon", "Nucleus Accumbens",
                           "Brainstem", "ILV Choroid Plexus",
                           "Cerebral WM Hypointensities", "Exterior",
                           "Unknown", "Posterior Superior Temporal Sulcus",
                           "Caudal Anterior Cingulate", "Premotor",
                           "Corpus Callosum", "Cuneus",
                           "Entorhinal Cortex", "Fusiform",
                           "Inferior Parietal", "Inferior Temporal",
                           "Isthmus Cingulate", "Lateral Occipital",
                           "Lateral Orbitofrontal", "Lingual",
                           "Medial Orbitofrontal", "Middle Temporal",
                           "Parahippocampal", "Paracentral",
                           "Pars Opercularis", "Pars Orbitalis",
                           "Pars Triangularis", "Pericalcarine",
                           "Primary Sensory", "Posterior Cingulate",
                           "Primary Motor", "Medial Parietal",
                           "Rostral Anterior Cingulate", "Anterior Middle Frontal",
                           "Superior Frontal", "Superior Parietal",
                           "Superior Temporal", "Supramarginal",
                           "Frontal Pole", "Temporal Pole",
                           "Transverse Temporal"]
        # Create 3 subplots for individual, comparison, and prediction charts
        self.a1 = fig.add_subplot(131)
        self.a1.set_title('Bar')
        self.a2 = fig.add_subplot(132, projection="polar")
        self.a2.set_title('Spie')
        self.lineax = self.a1.twinx()
        self.lineax.set_title("Bar")
        self.a3 = fig.add_subplot(133)
        self.a3.set_title("Scatter")
        self.texts = []
        # Add buttons & text boxes
        self.data_points = tk.Entry(root)
        self.data_points.pack(side="right", fill='both', expand=True)
        self.subject_num = tk.Entry(root)
        self.subject_num.pack(side="right", fill='both', expand=True)
        self.update = tk.Button(root,
                                text="Update",
                                command=self.update_plots)
        self.update.pack(side="right", fill='both', expand=True)
        self.clear_button = tk.Button(root,
                                      text="Clear Plots",
                                      command=self.clear_plots)
        self.clear_button.pack(side="right", fill='both', expand=True)
        self.current_patient = tk.Label(root,  # Text display of image 1 text
                                        text="ID:  Age:  DX: ",
                                        background="white")
        self.current_patient.pack(side="right", fill='both', expand=True)

        root.mainloop()

    def draw_bar(self, labels, widths, error):  # Draw the left bar graph
        # Clear the previous graph
        ax = self.a1
        ax.cla()
        self.lineax.cla()
        fig = self.fig
        subject = self.subject
        ax.set_axisbelow(True)
        # Space the structure labels according to the bar widths
        x_ticks = [sum(widths[:n]) + len(widths[:n]) for n in range(len(widths))]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels, fontsize=8)
        y_ticks = []
        y_labels = []
        # Label and the space the y-axis(Z scores)
        for n in range(9):
            z_score = n / 2 - 2
            y_labels.append(z_score)
            if z_score < 0:
                z_score = abs(z_score)
                y_ticks.append(Z_SCORE_TABLE.loc[int(100 * z_score), "Percentile"] * 100)
            else:
                y_ticks.append((1 - Z_SCORE_TABLE.loc[int(100 * z_score), "Percentile"]) * 100)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)
        # Find the relevant norm percentile data
        values = [subject[x + " Norm Percentile"] for x in labels]
        display_values = []
        colors = []
        for n, x in enumerate(labels): # Display PET scan data through color
            #Average the z_score
            z_score = (subject[x + " Left Z Score"] + subject[x + " Right Z Score"]) / 2
            #Limit the max and min color values
            sigmoid = (2 / (1 + np.exp(-1.3 * z_score))) - 1
            # Reverse the ventricle data for clarity
            if "Ventricle" in x:
                display_values.append(100 - values[n])
            else:
                display_values.append(values[n])
            if sigmoid > 0:
                colors.append([1, 0, 0, sigmoid])
            elif sigmoid < 0:
                colors.append([0, 0, 1, -sigmoid])
            else:
                colors.append([0, 0, 0, 0.5])
        #Draw the bar graph with error bars
        rects = ax.bar([x + sum(widths[:x]) for x in range(len(values))], display_values, widths, color=colors,
                       edgecolor=[x[:3] for x in colors], linewidth=1)
        errorbar = ax.errorbar([x + sum(widths[:x]) for x in range(len(values))], display_values,
                               yerr=error, linestyle="--", dashes=[0, 10])

        def autolabel(rects, xpos='center'):
            """
            Attach a text label above each bar in *rects*, displaying its height.

            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.
            """

            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0, 'right': 1, 'left': -1}

            for n, rect in enumerate(rects):
                height = rect.get_height()
                ax.annotate(int(values[n]),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                            textcoords="offset points",  # in both directions
                            ha=ha[xpos], va='bottom')
                ax.annotate(
                    str(round((subject[labels[n] + " Left Z Score"] + subject[labels[n] + " Right Z Score"]) / 2, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                    ha="center", va='top')

        autolabel(rects, xpos="left")
        # Limit the y-axis to 0 - 100(percentile)
        ax.set_ylim(0, 100)
        xmin, xmax = ax.get_xlim()
        # Set the x limits to the left edge of the first and the right of the last.
        ax.set_xlim(-widths[0] / 2, sum(widths) + len(widths) / 2)
        ax.set_xlabel("Structure")
        ax.set_ylabel("Percentile")
        # Create another graph on the same plot and draw marking lines for 5, 25, 50, 75, and 95th percentile
        lineax = self.lineax
        lineax.set_yticks([5, 25, 50, 75, 95])
        lineax.plot([0, 100], [95, 95], "--k", lw=1)
        lineax.plot([0, 100], [75, 75], "--k", lw=1)
        lineax.plot([0, 100], [50, 50], "k", lw=1)
        lineax.plot([0, 100], [25, 25], "--k", lw=1)
        lineax.plot([0, 100], [5, 5], "--k", lw=1)
        # Tilt the bottom labels at 45 degrees
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        self.canvas.draw()

    def draw_empty_spie(self, labels, texts):
        ax = self.a2
        fig = self.fig
        subject = self.subject
        ax.set_axisbelow(True)
        ax.tick_params(rotation="auto")
        # Remove y ticks and label the outer ones
        ax.set_yticklabels([])
        ax.set_xticklabels(labels, fontsize=8)
        # Find the relevant norm percentile data
        values = [subject[x + " Norm Percentile"] for x in labels]
        for n, value in enumerate(values):
            # Reverse the ventricle data for clarity
            if "Ventricle" in labels[n]:
                values[n] = 100 - values[n]
        xmin, xmax = ax.get_xlim()
        # Space outer ticks evenly across any number of structures
        ax.set_xticks(np.round(np.linspace(xmin, xmax, len(values) + 1), 2))
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
        width = 2 * np.pi / len(labels) * np.ones(len(labels))
        center_space = 150# Width of the center area
        # Draw the pie chart with the center empty
        ax.bar(angles, values, width=width, edgecolor='black', linewidth=1, bottom=center_space, alpha=0.3)
        patch = patches.Rectangle((0, 0), 2 * np.pi, center_space, color="w")
        ax.add_patch(patch)
        ax.set_ylim(0, 100 + center_space)

        for text in self.texts: # Clear previous text
            text.set_visible(False)
        for n, text in enumerate(texts): # Inputs a list of strings and displays each one on its own line
            self.texts.append(ax.text((8 + n) / 10 * np.pi, 100, text))
        # Draw 50th percentile line
        linex = np.linspace(0, 2 * np.pi, 500)
        liney = (center_space + 50) * np.ones(500)
        ax.plot(linex, liney, "--r", lw=1)
        self.canvas.draw()

    def draw_scatter(self):
        ax = self.a3
        ax.cla()
        fig = self.fig
        subject = self.subject
        # Graph the normal predicted vs. actual, then plot the subject's predicted vs. actual
        ax.scatter(NORMAL_DATA["Actual Age"], NORMAL_DATA["Predicted Age"])
        ax.scatter(TEST_DATA["Actual Age"].loc[int(self.subject_num.get())],
                   TEST_DATA["Predicted Age"].loc[int(self.subject_num.get())], color="red")
        self.canvas.draw()

    def update_plots(self):
        #FInd a subject's data with its label number
        self.subject = ALL_DATA.loc[int(self.subject_num.get())]
        #FInd the patient's ID, age, and health status
        patient_id = self.subject["Patient ID"]
        patient_age = round(self.subject["AGE"], 5)
        patient_dx = self.subject["DX"]
        raw_string = self.data_points.get()
        #split the requested data points on the commas
        points = raw_string.split(",")
        print(raw_string)
        print(points)
        #create the string to display in the center
        patient_string = "ID: " + patient_id + "  Age: " + str(patient_age) + "  DX: " + str(patient_dx)
        self.current_patient.config(text=patient_string)
        widths = range(1, len(points) + 1)
        #Draw all 3 graphs
        self.draw_bar(points, widths, widths)
        self.draw_empty_spie(points, ["ID: " + patient_id, "  Age: " + str(patient_age), "  DX: " + str(patient_dx)])
        self.draw_scatter()

    def clear_plots(self):
        #Remove all data from the graphs
        self.a1.cla()
        self.a2.cla()
        self.lineax.cla()
        self.canvas.draw()


app = Visualizer()
