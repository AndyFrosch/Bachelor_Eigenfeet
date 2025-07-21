import glob
import json
import os
import random
import csv

import tkinter as tk
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt

import numpy as np

from sklearn.metrics import auc, confusion_matrix, roc_curve

import matcher
import eigenfeet

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

ROOT_SINGLE = r'C:\Users\Andreas\Downloads\jku\bachelor\eigenfeet-biometrics\data\single'
ROOT_DOUBLE = r'C:\Users\Andreas\Downloads\jku\bachelor\eigenfeet-biometrics\data\double'


class PCAFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.root_directory = ROOT_SINGLE
        self.use_single_feet = tk.BooleanVar(value=True)
        self.angle = tk.BooleanVar(value=True)
        self.raw_dir = r''
        self.perprocessed_dir = r''
        self.pca_dir = r''
        
        tk.Label(self, text='PCA Calculation Page').grid(row=0, column=0, padx=27, pady=0)
        
        ##################
        # set diretories #
        ##################
        # directory selectors
        self.dir_check = tk.Checkbutton(self, text='Use singular feet images', variable=self.use_single_feet, command=self.toggle_single_double)
        self.dir_check.grid(row=1, column=0, padx=27, pady=0)

        self.angle_button = tk.Checkbutton(self, text="Use Rotational Alignment", variable=self.angle, command=self.toggle_angle)
        self.angle_button.grid(row=1, column=2, padx=5, pady=5)

        # directory selectors label
        self.dir_label = tk.Label(self, text=f'Root directory: ...{self.root_directory[-10:]}')
        self.dir_label.grid(row=3, column=0, padx=5, pady=5)
        ##################
        # set diretories #
        ##################

        

        #################
        # calculate pca #
        #################
        # # Entry box to input number of components
        tk.Label(self, text='Number of components:').grid(row=1, column=4)
        self.num_components_entry = tk.Entry(self)
        self.num_components_var = tk.StringVar(value="20")
        self.num_components_entry = tk.Entry(self, textvariable=self.num_components_var)
        self.num_components_entry.grid(row=1, column=5)

        # # # Button to trigger PCA calculation
        calc_button = tk.Button(self, text='Calculate PCA', command=self.calculate_pca)
        calc_button.grid(row=1, column=6)

        # # # Label to show status/results
        self.pca_status_label = tk.Label(self, text="")
        self.pca_status_label.grid(row=1, column=7)
        #################
        # calculate pca #
        #################

        self.load_config()
        self.toggle_single_double()
        self.toggle_angle()


    def toggle_single_double(self):
        if self.use_single_feet.get():
            self.root_directory = ROOT_SINGLE
            self.raw_dir = f'{self.root_directory}\\raw\\'
            self.perprocessed_dir = f'{self.root_directory}\\preprocessed\\'
            self.pca_dir = f'{self.root_directory}\\pca_components\\'

        else:
            self.root_directory = ROOT_DOUBLE
            self.raw_dir = f'{self.root_directory}\\raw\\'
            self.perprocessed_dir = f'{self.root_directory}\\preprocessed\\'
            self.pca_dir = f'{self.root_directory}\\pca_components\\'

        self.master.raw_dir = self.raw_dir
        self.master.use_single = self.use_single_feet.get()
        self.dir_label.config(text=f'Root directory: ...{self.root_directory[-10:]}')
        self.save_config()

    def toggle_angle(self):
        self.master.angle = self.angle.get()

    def select_directory(self):
        folder = filedialog.askdirectory()
        if folder:
            if self.use_single_feet:
                self.root_directory = f'{folder}\\single\\'
            else:
                self.root_directory = f'{folder}\\double\\'
            self.dir_label.config(text=f'Root directory: ...{self.root_directory[-10:]}')
            self.save_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                self.use_single_feet.set(config.get('use_single', True))
                self.root_directory = config.get('root_directory')
                self.dir_label.config(text=f'Root directory: ...{self.root_directory[-10:]}')


    def save_config(self):
        config = {
            "use_single": self.use_single_feet.get(),
            "root_directory": self.root_directory
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)


    def calculate_pca(self):
        num = self.num_components_entry.get()
        double_single = self.use_single_feet.get()
        angle = self.angle.get()
        self.pca_status_label.config(text=f"Calculating PCA with {num} components...")
        if num.strip():
            _pca, _x_pca, _eigenfeet, _x, _y = eigenfeet.start(self.raw_dir, self.perprocessed_dir, self.pca_dir, int(num), False, double_single=not double_single, angle=angle)
        else:
            _pca, _x_pca, _eigenfeet, _x, _y = eigenfeet.start(self.raw_dir, self.perprocessed_dir, self.pca_dir, skip_preprocessing=False, double_single=not double_single, angle=angle)
            
        print("calculation done")
        # After calculation, update status again
        self.pca_status_label.config(text="PCA calculation complete!")
        self.master._pca = _pca
        self.master._x_pca = _x_pca
        self.master._eigenfeet = _eigenfeet
        self.master._x = _x
        self.master._y = _y


    

class TestFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        tk.Label(self, text="Testing Page").grid(row=0, column=0, padx=27, pady=0)
        # Add testing widgets here



        # Entry box to input number of rounds
        tk.Label(self, text='Number of test runs:').grid(row=1, column=1)
        self.num_tests_entry = tk.Entry(self)
        self.num_tests_var = tk.StringVar(value="4")
        self.num_tests_entry = tk.Entry(self, textvariable=self.num_tests_var)
        self.num_tests_entry.grid(row=1, column=2)
        

        # # # Button to start tests
        self.test_button = tk.Button(self, text='Start Test', command=self.start_test)
        self.test_button.grid(row=1, column=3)


        self.results = tk.Label(self, text="Test results: ")
        self.results.grid(row=2, column=2, pady=50)

    def save_results_to_csv(self, file_path, results_dict):
        fieldnames = list(results_dict.keys())
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            writer.writerow(results_dict)


    def start_test(self):
        app = self.master
        _pca = app._pca
        _x_pca = app._x_pca
        _y = app._y

        test_root = app.raw_dir

        if not os.path.isdir(test_root):
            self.results.config(text="Invalid test directory.")
            return

        if _pca is None or _x_pca is None or _y is None:
            self.results.config(text="PCA must be calculated first.")
            return

        ident_true_labels = []
        ident_pred_labels = []

        verif_true = []  # 1 for genuine, 0 for impostor
        verif_dist = []  # distance
        verif_preds = []

        threshold = 80 


        for n in range(int(self.num_tests_var.get())):
            print(f"--- Test Loop {n + 1} ---")

            

            for user_folder in os.listdir(test_root):
                user_path = os.path.join(test_root, user_folder)
                if not os.path.isdir(user_path):
                    continue

                image_files = glob.glob(os.path.join(user_path, '*.jpg'))
                image_files += glob.glob(os.path.join(user_path, '*.jpeg'))
                image_files = [img for img in image_files if '-1' not in img]
                if not image_files:
                    continue
                test_image = random.choice(image_files)
                print(test_image)
                try:
                    # Preprocess and project
                    img_vec = matcher.preprocess_single_image(test_image, app.angle)
                    projected = matcher.project_image(_pca, img_vec)

                    true_label = user_folder
                    # ---------- IDENTIFICATION ----------
                    pred_label, _ = matcher.match_image(projected, _x_pca, _y)
                    ident_true_labels.append(true_label)
                    ident_pred_labels.append(pred_label)

                    # ---------- VERIFICATION ----------
                    # genuine attempt
                    genuine_result, genuine_dist = matcher.verify_identity(projected, true_label, _x_pca, _y, threshold)
                    verif_true.append(1)
                    verif_dist.append(genuine_dist)
                    verif_preds.append(int(genuine_result))

                    # impostor attempt
                    other_labels = list(set(_y) - {true_label})
                    fake_label = random.choice(other_labels)
                    impostor_result, impostor_dist = matcher.verify_identity(projected, fake_label, _x_pca, _y, threshold)
                    verif_true.append(0)
                    verif_dist.append(impostor_dist)
                    verif_preds.append(int(impostor_result))

                except Exception as e:
                    print(f"Error with file {test_image}: {e}")
                    continue

        # # Identification accuracy
        ident_acc = 100 * np.mean(np.array(ident_true_labels) == np.array(ident_pred_labels))
        # # Compute verification accuracy
        verif_acc = 100 * np.mean(np.array(verif_preds) == np.array(verif_true))
        

        # Confusion Matrix for Identification
        labels = sorted(set(_y))
        cm = confusion_matrix(ident_true_labels, ident_pred_labels, labels=labels)

        # Verification ROC Curve & AUC
        fpr, tpr, thresholds = roc_curve(verif_true, [-score for score in verif_dist])  # negate distances to have higher scores = positive
        roc_auc = auc(fpr, tpr)

        # Threshold tuning (choose threshold maximizing Youden's J = TPR - FPR)
        youdens_j = tpr - fpr
        optimal_idx = np.argmax(youdens_j)
        optimal_threshold = -thresholds[optimal_idx]

        # Compute verification accuracy with optimal threshold
        verif_preds = [1 if score <= optimal_threshold else 0 for score in verif_dist]
        verif_acc_optimal = 100 * np.mean(np.array(verif_preds) == np.array(verif_true))

        # ROC curve
        fig = plt.Figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot(111)

        # Compute FAR and FRR at the optimal threshold
        verif_true_array = np.array(verif_true)
        verif_preds_array = np.array(verif_preds)

        # FAR: Impostors accepted
        impostor_mask = verif_true_array == 0
        false_accepts = np.sum((verif_preds_array == 1) & impostor_mask)
        total_impostors = np.sum(impostor_mask)
        FAR = 100 * false_accepts / total_impostors if total_impostors else 0

        # FRR: Genuine users rejected
        genuine_mask = verif_true_array == 1
        false_rejects = np.sum((verif_preds_array == 0) & genuine_mask)
        total_genuine = np.sum(genuine_mask)
        FRR = 100 * false_rejects / total_genuine if total_genuine else 0

        # results
        # Plot ROC curve
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Verification ROC Curve')
        ax.legend(loc='lower right')

        if hasattr(self, 'roc_canvas'):
            self.roc_canvas.get_tk_widget().destroy()

        self.roc_canvas = FigureCanvasTkAgg(fig, master=self)
        self.roc_canvas.draw()
        self.roc_canvas.get_tk_widget().grid(row=2, column=7, sticky='nw')


        results_text = f"Identification Accuracy: {ident_acc:.2f}%\n"
        results_text += f"Verification Accuracy: {verif_acc:.2f}% with threshold {threshold}\n"
        results_text += f"Optimal Verification Threshold: {optimal_threshold:.2f}\n"
        results_text += f"Optimal Verification Accuracy: {verif_acc_optimal:.2f}%\n"
        results_text += f"Verification AUC: {roc_auc:.2f}\n"
        results_text += f"\nFalse Acceptance Rate (FAR): {FAR:.2f}%"
        results_text += f"\nFalse Rejection Rate (FRR): {FRR:.2f}%"
        results_text += "\nConfusion Matrix (Identification):\n" + str(cm)


        results_dict = {
            "Components": int(app.frames["PCAFrame"].num_components_var.get()),
            "Rotated": app.angle,
            "BothFeet": not app.use_single,
            "Ident_Accuracy": round(ident_acc, 2),
            "Verif_Accuracy": round(verif_acc_optimal, 2),
            "AUC": round(roc_auc, 2),
            "FAR": round(FAR, 2),
            "FRR": round(FRR, 2),
            "Threshold": round(optimal_threshold, 2),
        }

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'test_results.csv')
        self.save_results_to_csv(file_path, results_dict)

        self.results.config(text=results_text)

    

class MatchFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.input_ident = ''
        self.input_verify = ''
        self.matched_label = ''
        self.verific_result = ''
        self.distance = 0

        tk.Label(self, text="Matching Page").grid(row=0, column=0, padx=27, pady=0)
        tk.Label(self, text='').grid(row=1, column=2, padx=70)

        tk.Label(self, text='Identification:').grid(row=1, column=0)
        self.dir_select_button1 = tk.Button(self, text="Select User to identify", command=self.select_directory1)
        self.dir_select_button1.grid(row=2, column=0, padx=5)
        self.dir_label1 = tk.Label(self, text=f'Directory: ...{self.input_ident[-10:]}')
        self.dir_label1.grid(row=2, column=1, padx=5, pady=5)

        self.indentification_button = tk.Button(self, text="Identify", command=self.match_ident)
        self.indentification_button.grid(row=3, column=0, padx=5)

        self.ident_result = tk.Label(self, text=f'Result: ...{self.matched_label}')
        self.ident_result.grid(row=3, column=1, padx=5, pady=5)

        tk.Label(self, text='').grid(row=3, column=2, padx=70)

        tk.Label(self, text='Verification:').grid(row=1, column=3)
        self.dir_select_button2 = tk.Button(self, text="Select User to identify", command=self.select_directory2)
        self.dir_select_button2.grid(row=2, column=3)
        self.dir_label2 = tk.Label(self, text=f'Directory: ...{self.input_verify[-10:]}')
        self.dir_label2.grid(row=2, column=4, pady=5)

        tk.Label(self, text='Expected User (userx):').grid(row=1, column=5)
        self.expected_user_entry = tk.Entry(self)
        self.expected_user_var = tk.StringVar(value="")
        self.expected_user_entry = tk.Entry(self, textvariable=self.expected_user_var)
        self.expected_user_entry.grid(row=2, column=5)

        self.verification_button = tk.Button(self, text="Verify", command=self.match_verify)
        self.verification_button.grid(row=3, column=3, padx=5)

        self.verify_result = tk.Label(self, text=f'Result: ...{self.verific_result}')
        self.verify_result.grid(row=3, column=4, pady=5)


    def select_directory1(self):
        folder = filedialog.askopenfilename()
        if folder:
            self.input_ident = folder
            self.dir_label1.config(text=f'File: ...{self.input_ident[-10:]}')

    def select_directory2(self):
        folder = filedialog.askopenfilename()
        if folder:
            self.input_verify = folder
            self.dir_label2.config(text=f'File: ...{self.input_verify[-10:]}')

    def match_ident(self):
        # global _pca, _x_pca, _eigenfeet, _x, _y

        test_vector = matcher.project_image(self.master._pca, matcher.preprocess_single_image(self.input_ident, self.master.angle))
        self.matched_label, self.distance = matcher.match_image(test_vector, self.master._x_pca, self.master._y) # test_vector, database_vectors, labels
        self.ident_result.config(text=f'Result: {self.matched_label}')

    def match_verify(self):
        # global _pca, _x_pca, _eigenfeet, _x, _y

        flattened_img = matcher.preprocess_single_image(self.input_verify, self.master.angle)
        test_vector = matcher.project_image(self.master._pca, flattened_img)
        claimed_label = self.expected_user_entry.get()
        
        self.verific_result, self.distance = matcher.verify_identity(test_vector, claimed_label, self.master._x_pca, self.master._y, 80) #test_vector, claimed_label, database_vectors, labels, threshold
        self.verify_result.config(text=f'Result: {self.verific_result}')



class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Eigenfeet Matcher')
        self.geometry('880x700')

        # Shared PCA variables
        self._pca = None
        self._x = None
        self._x_pca = None
        self._y = None
        self._eigenfeet = None

        self.use_single = None
        self.angle = None
        self.raw_dir = None

        # Create frames
        self.frames = {}
        for F in (PCAFrame, TestFrame, MatchFrame):
            page_name = F.__name__
            frame = F(self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame('PCAFrame')

        # Navigation buttons
        nav_frame = tk.Frame(self)
        nav_frame.grid(row=1, column=0, sticky='ew')
        tk.Button(nav_frame, text='PCA', command=lambda: self.show_frame('PCAFrame')).grid(row=0, column=0)
        tk.Button(nav_frame, text='Match', command=lambda: self.show_frame('MatchFrame')).grid(row=0, column=1)
        tk.Button(nav_frame, text='Test', command=lambda: self.show_frame('TestFrame')).grid(row=0, column=2)

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()




if __name__ == '__main__':
    app = App()
    app.mainloop()

