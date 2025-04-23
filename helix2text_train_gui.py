"""
Helix2Text Training GUI Application

This script provides a graphical user interface for training the Helix2Text model
without using the terminal. It allows configuring training parameters, selecting data files,
and monitoring training progress.
"""

import os
import sys
import yaml
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch


# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Helix2Text.MetaModel.configs.default_config import default_config
from Helix2Text.MetaModel.scripts.train import train_model, load_config

class Helix2TextTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Helix2Text - Model Training")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Set up variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config_path = tk.StringVar(value="")
        self.train_data_path = tk.StringVar(value="")
        self.eval_data_path = tk.StringVar(value="")
        self.output_model_path = tk.StringVar(value="best_model.pt")
        self.use_wandb = tk.BooleanVar(value=True)
        self.project_name = tk.StringVar(value="Helix2Text")
        self.experiment_name = tk.StringVar(value=f"Training-{os.getpid()}")
        
        # Data splitting options
        self.split_data = tk.BooleanVar(value=True)
        self.train_data_fraction = tk.DoubleVar(value=1)
        self.train_split_fraction = tk.DoubleVar(value=0.9)
        
        # Training hyperparameters
        self.num_epochs = tk.IntVar(value=10)
        self.batch_size = tk.IntVar(value=8)
        self.learning_rate = tk.DoubleVar(value=5e-5)
        self.early_stop_patience = tk.IntVar(value=3)
        self.grad_accum_steps = tk.IntVar(value=4)
        
        # Create the main notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.config_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)
        self.metrics_tab = ttk.Frame(self.notebook)
        self.dataset_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.config_tab, text="Configuration")
        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.metrics_tab, text="Metrics")
        self.notebook.add(self.dataset_tab, text="Datasets")
        
        # Set up the Configuration tab
        self.setup_config_tab()
        
        # Set up the Training tab
        self.setup_training_tab()
        
        # Set up the Metrics tab
        self.setup_metrics_tab()

        # Set up the Dataset tab
        self.setup_dataset_tab() 
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Configure training parameters to begin.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Training state
        self.training_thread = None
        self.is_training = False
        self.stop_training = False
        self.training_metrics = {"loss": [], "bleu": []}
        
    def setup_config_tab(self):
        """Setup the configuration tab with all training options."""
        # Create frames for different sections
        paths_frame = ttk.LabelFrame(self.config_tab, text="Paths")
        paths_frame.pack(fill=tk.X, padx=10, pady=5)
        
        hyperparams_frame = ttk.LabelFrame(self.config_tab, text="Training Hyperparameters")
        hyperparams_frame.pack(fill=tk.X, padx=10, pady=5)
        
        wandb_frame = ttk.LabelFrame(self.config_tab, text="Weights & Biases Logging")
        wandb_frame.pack(fill=tk.X, padx=10, pady=5)
        
        data_split_frame = ttk.LabelFrame(self.config_tab, text="Data Splitting Options")
        data_split_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Paths section
        # Config file
        config_frame = ttk.Frame(paths_frame)
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Config File (Optional):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(config_frame, textvariable=self.config_path, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(config_frame, text="Browse...", command=self.browse_config).pack(side=tk.LEFT, padx=5)
        
        # Train data
        train_data_frame = ttk.Frame(paths_frame)
        train_data_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(train_data_frame, text="Training Data CSV:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(train_data_frame, textvariable=self.train_data_path, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(train_data_frame, text="Browse...", command=self.browse_train_data).pack(side=tk.LEFT, padx=5)
        
        # Eval data
        eval_data_frame = ttk.Frame(paths_frame)
        eval_data_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(eval_data_frame, text="Evaluation Data CSV:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(eval_data_frame, textvariable=self.eval_data_path, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(eval_data_frame, text="Browse...", command=self.browse_eval_data).pack(side=tk.LEFT, padx=5)
        
        # Output model path
        model_path_frame = ttk.Frame(paths_frame)
        model_path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_path_frame, text="Output Model Path:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(model_path_frame, textvariable=self.output_model_path, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="Browse...", command=self.browse_output_model).pack(side=tk.LEFT, padx=5)
        
        # Hyperparameters section
        # Use grid layout for better organization
        row = 0
        
        ttk.Label(hyperparams_frame, text="Number of Epochs:").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(hyperparams_frame, from_=1, to=100, textvariable=self.num_epochs, width=10).grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(hyperparams_frame, text="Batch Size:").grid(row=row, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(hyperparams_frame, from_=1, to=64, textvariable=self.batch_size, width=10).grid(row=row, column=3, padx=5, pady=5, sticky=tk.W)
        
        row += 1
        ttk.Label(hyperparams_frame, text="Learning Rate:").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(hyperparams_frame, textvariable=self.learning_rate, width=10).grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(hyperparams_frame, text="Early Stop Patience:").grid(row=row, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(hyperparams_frame, from_=1, to=10, textvariable=self.early_stop_patience, width=10).grid(row=row, column=3, padx=5, pady=5, sticky=tk.W)
        
        row += 1
        ttk.Label(hyperparams_frame, text="Gradient Accumulation Steps:").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(hyperparams_frame, from_=1, to=16, textvariable=self.grad_accum_steps, width=10).grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Add some space between rows
        for i in range(3):
            hyperparams_frame.grid_columnconfigure(i, weight=1)
        
        # W&B logging section
        wandb_check = ttk.Checkbutton(wandb_frame, text="Enable Weights & Biases Logging", variable=self.use_wandb)
        wandb_check.pack(anchor=tk.W, padx=5, pady=5)
        
        wandb_details_frame = ttk.Frame(wandb_frame)
        wandb_details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(wandb_details_frame, text="Project Name:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(wandb_details_frame, textvariable=self.project_name, width=20).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(wandb_details_frame, text="Experiment Name:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(wandb_details_frame, textvariable=self.experiment_name, width=20).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Data splitting section
        split_check = ttk.Checkbutton(data_split_frame, text="Enable Data Splitting", variable=self.split_data)
        split_check.pack(anchor=tk.W, padx=5, pady=5)
        
        split_details_frame = ttk.Frame(data_split_frame)
        split_details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(split_details_frame, text="Train Data Fraction:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(split_details_frame, textvariable=self.train_data_fraction, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(split_details_frame, text="Train Split Fraction:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(split_details_frame, textvariable=self.train_split_fraction, width=10).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Config preview box
        preview_frame = ttk.LabelFrame(self.config_tab, text="Configuration Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.config_preview = scrolledtext.ScrolledText(preview_frame, height=10)
        self.config_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom buttons
        button_frame = ttk.Frame(self.config_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Load Config", command=self.load_config_from_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Update Preview", command=self.update_config_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config_to_file).pack(side=tk.LEFT, padx=5)
        
        # Initial config preview
        self.update_config_preview()
    
    def setup_training_tab(self):
        """Setup the training tab with control buttons and logs."""
        # Controls frame
        controls_frame = ttk.Frame(self.training_tab)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_button = ttk.Button(controls_frame, text="Start Training", command=self.start_training)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(controls_frame, text="Stop Training", command=self.stop_training_process, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.training_tab, text="Training Progress")
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Not started")
        self.progress_label.pack(anchor=tk.W, padx=5)
        
        # Live metrics frame
        metrics_frame = ttk.LabelFrame(self.training_tab, text="Current Training Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.current_metrics = {}
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # Create metric labels (will be updated during training)
        row = 0
        col = 0
        for i, metric in enumerate(['Epoch', 'Loss', 'BLEU', 'Learning Rate', 'ETA']):
            ttk.Label(metrics_grid, text=f"{metric}:").grid(row=row, column=col*2, padx=5, pady=2, sticky=tk.W)
            label = ttk.Label(metrics_grid, text="-")
            label.grid(row=row, column=col*2+1, padx=5, pady=2, sticky=tk.W)
            self.current_metrics[metric.lower().replace(' ', '_')] = label
            
            col += 1
            if col > 2:  # 3 columns of metrics
                col = 0
                row += 1
        
        # Training log
        log_frame = ttk.LabelFrame(self.training_tab, text="Training Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add buttons to save logs
        button_frame = ttk.Frame(self.training_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Log", command=self.save_log).pack(side=tk.LEFT, padx=5)
    
    def setup_metrics_tab(self):
        """Setup the metrics tab with charts for visualizing training progress."""
        # Create frame for plots
        plots_frame = ttk.Frame(self.metrics_tab)
        plots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure with subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(7, 8))
        
        # Set up the loss plot
        self.ax1.set_title('Training Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.loss_line, = self.ax1.plot([], [], 'b-', marker='o', label='Training Loss')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Set up the BLEU score plot
        self.ax2.set_title('BLEU Score')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('BLEU')
        self.bleu_line, = self.ax2.plot([], [], 'g-', marker='o', label='BLEU Score')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Embed the plots in the tkinter frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add export buttons
        export_frame = ttk.Frame(self.metrics_tab)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(export_frame, text="Export Metrics CSV", command=self.export_metrics_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Save Charts", command=self.save_charts).pack(side=tk.LEFT, padx=5)

    def setup_dataset_tab(self):
        """Setup the dataset tab for managing datasets."""
        dataset_frame = ttk.LabelFrame(self.dataset_tab, text="Dataset Management")
        dataset_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Available datasets section
        ttk.Label(dataset_frame, text="Available Datasets:", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        
        # Protein to DNA Dataset frame
        protein_dna_frame = ttk.Frame(dataset_frame)
        protein_dna_frame.pack(fill=tk.X, padx=5, pady=5)
        
        dataset_desc = ttk.Label(protein_dna_frame, text="Protein to DNA Sequence Descriptions\nA curated dataset of protein descriptions mapped to their corresponding DNA sequences.", wraplength=600, justify=tk.LEFT)
        dataset_desc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Download location
        self.download_path = tk.StringVar(value=os.getcwd())
        
        # Download path selection
        path_frame = ttk.Frame(dataset_frame)
        path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(path_frame, text="Download Location:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(path_frame, textvariable=self.download_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="Browse...", command=self.browse_download_path).pack(side=tk.LEFT, padx=5)
        
        # Download button
        ttk.Button(protein_dna_frame, text="Transfer Dataset", command=self.download_protein_dna_dataset).pack(side=tk.RIGHT, padx=5)
        
        # Download progress
        self.download_progress_frame = ttk.Frame(dataset_frame)
        self.download_progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.download_progress = ttk.Progressbar(self.download_progress_frame, orient="horizontal", length=100, mode="determinate")
        self.download_progress.pack(fill=tk.X, padx=5, pady=5)
        
        self.download_label = ttk.Label(self.download_progress_frame, text="")
        self.download_label.pack(anchor=tk.W, padx=5)
        
        # Dataset information section
        info_frame = ttk.LabelFrame(dataset_frame, text="Dataset Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.dataset_info = scrolledtext.ScrolledText(info_frame, height=10)
        self.dataset_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.dataset_info.insert(tk.END, "Dataset Information:\n\nName: Protein to DNA Sequence Dataset\nSource: Rsync Local Fetch\nFormat: CSV\n\nThis dataset contains protein descriptions and their corresponding DNA sequences, suitable for training the Helix2Text model to generate text descriptions from DNA sequences and vice versa.")
        self.dataset_info.config(state=tk.DISABLED)

    def browse_download_path(self):
        """Browse for the download location directory."""
        path = filedialog.askdirectory(
            initialdir=self.download_path.get(),
            title="Select Download Location"
        )
        if path:
            self.download_path.set(path)
    
    def browse_config(self):
        """Browse for a config YAML file."""
        path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Config File",
            filetypes=(("YAML files", "*.yaml *.yml"), ("All files", "*.*"))
        )
        if path:
            self.config_path.set(path)
    
    def browse_train_data(self):
        """Browse for training data CSV file."""
        path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Training Data CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if path:
            self.train_data_path.set(path)
    
    def browse_eval_data(self):
        """Browse for evaluation data CSV file."""
        path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Evaluation Data CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if path:
            self.eval_data_path.set(path)
    
    def browse_output_model(self):
        """Browse for output model path."""
        path = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            title="Save Model As",
            filetypes=(("PyTorch Models", "*.pt *.pth"), ("All files", "*.*")),
            defaultextension=".pt"
        )
        if path:
            self.output_model_path.set(path)
    
    def update_config_preview(self):
        """Update the configuration preview text box with current settings."""
        config = self.get_config_dict()
        
        # Format as YAML style
        preview_text = "# Helix2Text Training Configuration\n\n"
        
        for section, values in config.items():
            preview_text += f"{section}:\n"
            if isinstance(values, dict):
                for key, value in values.items():
                    preview_text += f"  {key}: {value}\n"
            else:
                # Handle case where values is not a dictionary (e.g., a string)
                preview_text += f"  {values}\n"
            preview_text += "\n"
        
        # Update preview
        self.config_preview.delete("1.0", tk.END)
        self.config_preview.insert(tk.END, preview_text)
    
    def get_config_dict(self):
        """Create a configuration dictionary from GUI values."""
        config = {
            "project_name": self.project_name.get(),
            "experiment_name": self.experiment_name.get(),
            "training": {
                "num_epochs": self.num_epochs.get(),
                "batch_size": self.batch_size.get(),
                "learning_rate": self.learning_rate.get(),
                "early_stop_patience": self.early_stop_patience.get(),
                "grad_accum_steps": self.grad_accum_steps.get(),
                "use_wandb": self.use_wandb.get()
            },
            "data": {
                "train_file": self.train_data_path.get(),
                "eval_file": self.eval_data_path.get(),
                "split_data": self.split_data.get(),
                "train_data_fraction": self.train_data_fraction.get(),
                "train_split_fraction": self.train_split_fraction.get()
            },
            "model": {
                "output_path": self.output_model_path.get()
            }
        }
        return config
    
    def get_config_object(self):
        """Create a configuration object that matches the expected format."""
        # Start with default configuration
        config = default_config
        
        # Update with GUI values
        config.project_name = self.project_name.get()
        config.experiment_name = self.experiment_name.get()
        
        # Update training configuration
        config.training.num_epochs = self.num_epochs.get()
        config.training.batch_size = self.batch_size.get()
        config.training.learning_rate = self.learning_rate.get()
        config.training.early_stop_patience = self.early_stop_patience.get()
        config.training.grad_accum_steps = self.grad_accum_steps.get()
        
        # Update data configuration
        if self.train_data_path.get():
            config.data.train_file = self.train_data_path.get()
        if self.eval_data_path.get():
            config.data.eval_file = self.eval_data_path.get()
        config.data.split_data = self.split_data.get()
        config.data.train_data_fraction = self.train_data_fraction.get()
        config.data.train_split_fraction = self.train_split_fraction.get()
        
        # Update model output path
        config.model.output_path = self.output_model_path.get()
        
        return config
    
    def load_config_from_file(self):
        """Load configuration from a YAML file."""
        config_path = self.config_path.get()
        if not config_path:
            messagebox.showwarning("Warning", "No config file selected.")
            return
        
        try:
            config = load_config(config_path)
            
            # Update GUI elements with loaded config
            self.num_epochs.set(config.training.num_epochs)
            self.batch_size.set(config.training.batch_size)
            self.learning_rate.set(config.training.learning_rate)
            self.early_stop_patience.set(config.training.early_stop_patience)
            self.grad_accum_steps.set(config.training.grad_accum_steps)
            
            if hasattr(config, "project_name"):
                self.project_name.set(config.project_name)
            if hasattr(config, "experiment_name"):
                self.experiment_name.set(config.experiment_name)
            
            if hasattr(config.data, "train_file"):
                self.train_data_path.set(config.data.train_file)
            if hasattr(config.data, "eval_file"):
                self.eval_data_path.set(config.data.eval_file)
            if hasattr(config.data, "split_data"):
                self.split_data.set(config.data.split_data)
            if hasattr(config.data, "train_data_fraction"):
                self.train_data_fraction.set(config.data.train_data_fraction)
            if hasattr(config.data, "train_split_fraction"):
                self.train_split_fraction.set(config.data.train_split_fraction)
                
            # Update preview
            self.update_config_preview()
            
            messagebox.showinfo("Success", f"Config loaded from {config_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {str(e)}")
    
    def download_protein_dna_dataset(self):
        """Guide the user to transfer the protein to DNA sequence dataset from their local PC using rsync."""
        # Create rsync dialog
        rsync_dialog = tk.Toplevel(self.root)
        rsync_dialog.title("Transfer Dataset from Local PC")
        rsync_dialog.geometry("650x500")
        rsync_dialog.grab_set()  # Make the dialog modal
        
        # Create instructions frame
        instructions_frame = ttk.LabelFrame(rsync_dialog, text="Instructions")
        instructions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        instructions = scrolledtext.ScrolledText(instructions_frame, height=12, wrap=tk.WORD)
        instructions.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add instructions text
        instructions.insert(tk.END, r"""Follow these steps to transfer your dataset from your local PC to this server:
                            
        Path: \Users\lhfor\AppData\Local\Programs\Python\Python38-32\Helix2TextDataRetreival\Dataset\protein_to_dna_sequence_descriptions.csv

        1. This feature generates an rsync command for you to run on your local PC.

        2. Copy the generated command and run it in your local terminal/command prompt.

        3. It will securely transfer your dataset to this server with progress tracking.

        4. After transferring is complete, check for the file and register it for training.

        Note: rsync must be installed on your local PC:
        - Mac/Linux: Included by default
        - Windows: Install cwrsync, use WSL, or Git Bash
        """)

        instructions.config(state=tk.DISABLED)
        
        # Input frame for local file details
        input_frame = ttk.LabelFrame(rsync_dialog, text="File Details")
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Local file path
        local_path_var = tk.StringVar()
        local_path_frame = ttk.Frame(input_frame)
        local_path_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(local_path_frame, text="Path to dataset on your local PC:").pack(anchor=tk.W, padx=5, pady=5)
        ttk.Entry(local_path_frame, textvariable=local_path_var, width=50).pack(fill=tk.X, padx=5)
        ttk.Label(local_path_frame, text="Example: /home/user/datasets/protein_to_dna.csv or C:\\Data\\dataset.csv").pack(anchor=tk.W, padx=5)
        
        # Get server details
        server_username = os.environ.get('USER', os.environ.get('USERNAME', 'username'))
        server_hostname = self.get_hostname()
        server_path = self.download_path.get()
        
        # Generate command frame
        cmd_frame = ttk.LabelFrame(rsync_dialog, text="Rsync Command")
        cmd_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        cmd_text = scrolledtext.ScrolledText(cmd_frame, height=6, wrap=tk.WORD)
        cmd_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        def update_rsync_command():
            """Update the rsync command based on input."""
            local_path = local_path_var.get().strip()
            if not local_path:
                cmd_text.delete("1.0", tk.END)
                cmd_text.insert(tk.END, "Please enter the path to your dataset file on your local PC.")
                return
            
            # Adjust path format for Windows vs Unix
            if '\\' in local_path:
                # Convert Windows path to format suitable for rsync
                local_path = local_path.replace('\\', '/')
            
            # Build the rsync command
            rsync_cmd = f"rsync -avz --progress \"{local_path}\" {server_username}@{server_hostname}:{server_path}"
            
            cmd_text.delete("1.0", tk.END)
            cmd_text.insert(tk.END, rsync_cmd)
        
        # Add update button
        ttk.Button(local_path_frame, text="Generate Command", 
                  command=update_rsync_command).pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Status checker 
        status_frame = ttk.LabelFrame(rsync_dialog, text="Transfer Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(status_frame, text="After running the command on your local PC, click below to check for the file:").pack(anchor=tk.W, padx=5, pady=5)
        
        def check_file_exists():
            """Check if the file has been transferred."""
            local_path = local_path_var.get().strip()
            if not local_path:
                messagebox.showwarning("Warning", "Please enter the path to your dataset file.")
                return
                
            # Extract filename from path
            filename = os.path.basename(local_path.replace('\\', '/'))
            target_path = os.path.join(server_path, filename)
            
            if os.path.exists(target_path):
                # File exists, offer to use it
                try:
                    import pandas as pd
                    df = pd.read_csv(target_path)
                    row_count = len(df)
                    col_count = len(df.columns)
                    
                    # Close dialog
                    rsync_dialog.destroy()
                    
                    # Update UI with file info
                    self.download_label.config(
                        text=f"File received! {filename} ({row_count} rows, {col_count} columns)"
                    )
                    self.status_var.set(f"Dataset transfer complete: {target_path}")
                    
                    # Update dataset info
                    self.update_dataset_info(df, target_path)
                    
                    # Set as training data path if it's empty
                    if not self.train_data_path.get():
                        self.train_data_path.set(target_path)
                        
                    messagebox.showinfo(
                        "Success", 
                        f"Dataset transfer complete!\n\n"
                        f"File: {filename}\n"
                        f"Rows: {row_count}\n"
                        f"Columns: {col_count}\n\n"
                        "The dataset path has been set as the training data path."
                    )
                except Exception as e:
                    messagebox.showerror("Error", f"File found but appears to be invalid: {str(e)}")
            else:
                messagebox.showinfo("File Not Found", 
                                 f"The file {filename} has not been transferred yet.\n\n"
                                 "Please run the rsync command on your local PC first.")
        
        check_buttons = ttk.Frame(status_frame)
        check_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(check_buttons, text="Check for File", 
                  command=check_file_exists).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(check_buttons, text="Close", 
                  command=rsync_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Copy to clipboard button
        def copy_to_clipboard():
            """Copy the rsync command to clipboard."""
            cmd = cmd_text.get("1.0", tk.END).strip()
            rsync_dialog.clipboard_clear()
            rsync_dialog.clipboard_append(cmd)
            messagebox.showinfo("Copied", "Command copied to clipboard!")
        
        ttk.Button(cmd_frame, text="Copy to Clipboard", 
                  command=copy_to_clipboard).pack(anchor=tk.E, padx=5, pady=5)
    
    def update_dataset_info(self, df, file_path):
        """Update the dataset information text box with detailed information."""
        # Enable editing
        self.dataset_info.config(state=tk.NORMAL)
        
        # Clear current content
        self.dataset_info.delete("1.0", tk.END)
        
        # Add dataset information
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        
        info_text = f"Dataset Information:\n\n"
        info_text += f"Name: Protein to DNA Sequence Descriptions\n"
        info_text += f"Path: {file_path}\n"
        info_text += f"Size: {file_size:.2f} MB\n"
        info_text += f"Rows: {len(df)}\n"
        info_text += f"Columns: {len(df.columns)}\n\n"
        
        # Add column information
        info_text += "Columns:\n"
        for col in df.columns:
            info_text += f"- {col}\n"
        
        info_text += "\nSample Data:\n"
        sample_data = df.head(3).to_string()
        info_text += sample_data
        
        # Insert the text
        self.dataset_info.insert(tk.END, info_text)
        
        # Disable editing
        self.dataset_info.config(state=tk.DISABLED)

    def save_config_to_file(self):
        """Save current configuration to a YAML file."""
        config_dict = self.get_config_dict()
        
        file_path = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            title="Save Config As",
            filetypes=(("YAML files", "*.yaml"), ("All files", "*.*")),
            defaultextension=".yaml"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
                messagebox.showinfo("Success", f"Config saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config: {str(e)}")
    
    def start_training(self):
        """Begin the training process."""
        # Check if already training
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress.")
            return
        
        # Validate inputs
        if not self.validate_training_inputs():
            return
        
        # Update UI
        self.is_training = True
        self.stop_training = False
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress["value"] = 0
        self.clear_log()
        self.status_var.set("Training started...")
        
        # Reset metrics tracking
        self.training_metrics = {"loss": [], "bleu": []}
        
        # Get configuration
        config = self.get_config_object()
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.run_training, args=(config,))
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def run_training(self, config):
        """Execute the training process."""
        try:
            # Create custom handlers to intercept training progress
            class LogCallback:
                def __init__(self, gui):
                    self.gui = gui
                    self.epoch = 0
                    self.total_epochs = config.training.num_epochs
                
                def on_epoch_begin(self, epoch):
                    self.epoch = epoch
                    self.gui.log(f"\n[Epoch {epoch+1}/{self.total_epochs}] Starting...\n")
                    self.gui.update_progress((epoch / self.total_epochs) * 100)
                
                def on_epoch_end(self, epoch, metrics):
                    # Update metrics
                    avg_loss = metrics.get("avg_loss", 0)
                    bleu = metrics.get("bleu_score", 0)
                    self.gui.training_metrics["loss"].append(avg_loss)
                    self.gui.training_metrics["bleu"].append(bleu)
                    
                    # Log epoch results
                    self.gui.log(f"[Epoch {epoch+1}/{self.total_epochs}] " 
                                 f"Loss: {avg_loss:.4f} | BLEU: {bleu:.4f}")
                    
                    # Update progress bar and metrics display
                    self.gui.update_progress(((epoch + 1) / self.total_epochs) * 100)
                    self.gui.update_metrics(epoch + 1, avg_loss, bleu)
                    
                    # Update plots
                    self.gui.update_plots()
                    
                    # Check if we should stop
                    return self.gui.stop_training
                
                def on_step_end(self, step, step_metrics):
                    if step % 10 == 0:  # Log only every 10 steps to avoid flooding
                        loss = step_metrics.get("step_loss", 0)
                        lr = step_metrics.get("learning_rate", 0)
                        eta = step_metrics.get("eta_min", None)
                        
                        # Update UI elements
                        self.gui.current_metrics["loss"].config(text=f"{loss:.4f}")
                        self.gui.current_metrics["learning_rate"].config(text=f"{lr:.6f}")
                        if eta is not None:
                            self.gui.current_metrics["eta"].config(text=f"{eta:.1f} min")
            
            # Create a callback instance
            callback = LogCallback(self)
            
            # Disable wandb if not requested
            if not self.use_wandb.get():
                # Set the environment variable to use a dummy wandb implementation
                os.environ["WANDB_MODE"] = "disabled"
            
            # Add our custom callback to config
            config.callbacks = {"training": callback}
            
            self.log("Initializing training...")
            self.log(f"Using device: {self.device}")
            
            # Start training with the callback
            model, metrics = train_model(config)
            
            # Log final results
            self.log("\n============= Training Complete =============")
            self.log(f"Final BLEU Score: {metrics['bleu_score']:.4f}")
            self.log(f"Model saved to: {self.output_model_path.get()}")
            
            # Update UI
            self.status_var.set("Training completed successfully!")
            messagebox.showinfo("Success", "Training completed successfully!")
        
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.status_var.set(f"Training failed: {str(e)}")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
        
        finally:
            # Reset UI state
            self.is_training = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def stop_training_process(self):
        """Signal the training process to stop at the next epoch."""
        if not self.is_training:
            return
            
        if messagebox.askyesno("Confirm", "Stop training? This will complete the current epoch and then stop."):
            self.stop_training = True
            self.status_var.set("Stopping training after current epoch...")
            self.log("\n[INFO] Training stop requested. Will stop after current epoch completes.")
    
    def validate_training_inputs(self):
        """Validate that all required inputs are provided."""
        # Check for training data
        if not self.train_data_path.get():
            messagebox.showerror("Error", "Training data file is required.")
            return False
        
        if not os.path.exists(self.train_data_path.get()):
            messagebox.showerror("Error", f"Training data file not found: {self.train_data_path.get()}")
            return False
        
        # Check for eval data
        if self.eval_data_path.get() and not os.path.exists(self.eval_data_path.get()):
            messagebox.showerror("Error", f"Evaluation data file not found: {self.eval_data_path.get()}")
            return False
        
        # Check hyperparameters
        try:
            if self.num_epochs.get() <= 0:
                messagebox.showerror("Error", "Number of epochs must be positive.")
                return False
                
            if self.batch_size.get() <= 0:
                messagebox.showerror("Error", "Batch size must be positive.")
                return False
                
            if self.learning_rate.get() <= 0:
                messagebox.showerror("Error", "Learning rate must be positive.")
                return False
        
        except Exception:
            messagebox.showerror("Error", "Invalid numeric values in hyperparameters.")
            return False
            
        return True
    
    def log(self, message):
        """Add a message to the log text widget."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # Scroll to the end
        self.root.update_idletasks()  # Update the UI
    
    def clear_log(self):
        """Clear the log text widget."""
        self.log_text.delete("1.0", tk.END)
    
    def save_log(self):
        """Save the training log to a file."""
        log_text = self.log_text.get("1.0", tk.END)
        
        file_path = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            title="Save Log As",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
            defaultextension=".txt"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(log_text)
                messagebox.showinfo("Success", f"Log saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save log: {str(e)}")
    
    def update_progress(self, value):
        """Update the progress bar."""
        self.progress["value"] = value
        progress_text = f"Progress: {value:.1f}%"
        self.progress_label.config(text=progress_text)
        self.root.update_idletasks()  # Update the UI
    
    def update_metrics(self, epoch, loss, bleu):
        """Update the metrics display."""
        self.current_metrics["epoch"].config(text=f"{epoch}")
        self.current_metrics["loss"].config(text=f"{loss:.4f}")
        self.current_metrics["bleu"].config(text=f"{bleu:.4f}")
    
    def update_plots(self):
        """Update the training metrics plots."""
        epochs = list(range(1, len(self.training_metrics["loss"])+1))
        
        # Update loss plot
        self.loss_line.set_data(epochs, self.training_metrics["loss"])
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update BLEU plot
        self.bleu_line.set_data(epochs, self.training_metrics["bleu"])
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Redraw canvas
        self.canvas.draw()
    
    def export_metrics_csv(self):
        """Export training metrics to a CSV file."""
        if not self.training_metrics["loss"]:
            messagebox.showwarning("Warning", "No training metrics to export.")
            return
            
        file_path = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            title="Save Metrics As",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            defaultextension=".csv"
        )
        
        if file_path:
            try:
                epochs = list(range(1, len(self.training_metrics["loss"])+1))
                
                with open(file_path, 'w', newline='') as f:
                    f.write("epoch,loss,bleu\n")
                    for i, (loss, bleu) in enumerate(zip(
                            self.training_metrics["loss"], 
                            self.training_metrics["bleu"])):
                        f.write(f"{epochs[i]},{loss},{bleu}\n")
                        
                messagebox.showinfo("Success", f"Metrics saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save metrics: {str(e)}")
    
    def save_charts(self):
        """Save the metrics charts as an image."""
        if not self.training_metrics["loss"]:
            messagebox.showwarning("Warning", "No training metrics to display in charts.")
            return
            
        file_path = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            title="Save Charts As",
            filetypes=(("PNG files", "*.png"), ("All files", "*.*")),
            defaultextension=".png"
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Charts saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save charts: {str(e)}")

    def get_hostname(self):
        """Get the current server's hostname."""
        try:
            import socket
            return socket.gethostname()
        except:
            return "server"

if __name__ == "__main__":
    root = tk.Tk()
    app = Helix2TextTrainingGUI(root)
    root.mainloop()