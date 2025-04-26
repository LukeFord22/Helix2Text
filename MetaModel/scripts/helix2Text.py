"""
Helix2Text GUI Application

This script provides a graphical user interface for using the Helix2Text model
to generate text from DNA sequences and vice versa without using the terminal.
"""

import os
import sys
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from threading import Thread
import pandas as pd
import requests  # Added for downloading files
from pathlib import Path  # Added for path handling

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change the import to use default_config variable instead of the non-existent load_config function
from Helix2Text.MetaModel.configs.default_config import default_config
from Helix2Text.MetaModel.models.dna_text_fusion import load_models, DNATextFusion

class Helix2TextGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Helix2Text - DNA-Text Generator")
        self.root.geometry("800x650")
        self.root.resizable(True, True)
        
        # Set up variables
        self.model = None
        self.dna_tokenizer = None
        self.text_tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = tk.StringVar(value="best_model.pt")
        self.max_length = tk.IntVar(value=128)
        self.mode = tk.StringVar(value="dna2text")
        
        # Create the main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create tabs
        self.single_tab = ttk.Frame(self.notebook)
        self.batch_tab = ttk.Frame(self.notebook)
        self.dataset_tab = ttk.Frame(self.notebook)  # Added dataset tab
        
        self.notebook.add(self.single_tab, text="Single Generation")
        self.notebook.add(self.batch_tab, text="Batch Processing")
        self.notebook.add(self.dataset_tab, text="Datasets")  # Added dataset tab
        
        # Set up the Single Generation tab
        self.setup_single_tab()
        
        # Set up the Batch Processing tab
        self.setup_batch_tab()
        
        # Set up the Dataset Management tab
        self.setup_dataset_tab()  # Added dataset tab setup
        
        # Status bar at the bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Select model and input to begin.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Load model button
        load_model_frame = ttk.Frame(main_frame)
        load_model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(load_model_frame, text="Model Path:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(load_model_frame, textvariable=self.model_path, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_model_frame, text="Browse...", command=self.browse_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_model_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        # Model info
        self.model_info_var = tk.StringVar(value="Model not loaded")
        ttk.Label(main_frame, textvariable=self.model_info_var).pack(anchor=tk.W, pady=5)
        
    def setup_single_tab(self):
        # Input section
        input_frame = ttk.LabelFrame(self.single_tab, text="Input")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Mode selection (DNA to Text or Text to DNA)
        mode_frame = ttk.Frame(input_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="Generation Mode:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="DNA to Text", variable=self.mode, value="dna2text").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Text to DNA", variable=self.mode, value="text2dna").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(mode_frame, text="Max Length:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Spinbox(mode_frame, from_=10, to=1000, textvariable=self.max_length, width=5).pack(side=tk.LEFT)
        
        # Input text
        ttk.Label(input_frame, text="Input:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.input_text = scrolledtext.ScrolledText(input_frame, height=6)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Generate button
        ttk.Button(input_frame, text="Generate", command=self.generate_single).pack(pady=10)
        
        # Output section
        output_frame = ttk.LabelFrame(self.single_tab, text="Output")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=8)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons for copying output
        button_frame = ttk.Frame(output_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Copy Output", command=self.copy_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Output", command=self.save_output).pack(side=tk.LEFT, padx=5)
    
    def setup_batch_tab(self):
        # Input section
        input_frame = ttk.LabelFrame(self.batch_tab, text="Batch Processing")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Mode selection for batch processing
        mode_frame = ttk.Frame(input_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.batch_mode = tk.StringVar(value="dna2text")
        ttk.Label(mode_frame, text="Batch Mode:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="DNA to Text", variable=self.batch_mode, value="dna2text").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Text to DNA", variable=self.batch_mode, value="text2dna").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(mode_frame, text="Max Length:").pack(side=tk.LEFT, padx=(20, 5))
        self.batch_max_length = tk.IntVar(value=128)
        ttk.Spinbox(mode_frame, from_=10, to=1000, textvariable=self.batch_max_length, width=5).pack(side=tk.LEFT)
        
        # File selection
        file_frame = ttk.Frame(input_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        self.input_file = tk.StringVar()
        ttk.Label(file_frame, text="Input CSV File:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(file_frame, textvariable=self.input_file, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse...", command=self.browse_input_file).pack(side=tk.LEFT, padx=5)
        
        # Output file selection
        output_file_frame = ttk.Frame(input_frame)
        output_file_frame.pack(fill=tk.X, pady=5)
        
        self.output_file = tk.StringVar(value="generation_results.csv")
        ttk.Label(output_file_frame, text="Output CSV File:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(output_file_frame, textvariable=self.output_file, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_file_frame, text="Browse...", command=self.browse_output_file).pack(side=tk.LEFT, padx=5)
        
        # Process button
        ttk.Button(input_frame, text="Process Batch", command=self.process_batch).pack(pady=10)
        
        # Progress section
        progress_frame = ttk.LabelFrame(self.batch_tab, text="Progress")
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to process")
        self.progress_label.pack(anchor=tk.W, padx=5)
        
        # Progress log
        ttk.Label(progress_frame, text="Processing Log:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.progress_log = scrolledtext.ScrolledText(progress_frame, height=8)
        self.progress_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_dataset_tab(self):
        """Set up the Dataset Management tab."""
        dataset_frame = ttk.LabelFrame(self.dataset_tab, text="Dataset Management")
        dataset_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(dataset_frame, text="Manage your datasets here.").pack(anchor=tk.W, padx=5, pady=5)
        # Additional dataset management functionality can be added here.
    
    def browse_model(self):
        path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Model File",
            filetypes=(("PyTorch Models", "*.pt *.pth"), ("All files", "*.*"))
        )
        if path:
            self.model_path.set(path)
    
    def browse_input_file(self):
        path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Input CSV File",
            filetypes=(("CSV Files", "*.csv"), ("All files", "*.*"))
        )
        if path:
            self.input_file.set(path)
    
    def browse_output_file(self):
        path = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            title="Save Results As",
            filetypes=(("CSV Files", "*.csv"), ("All files", "*.*")),
            defaultextension=".csv"
        )
        if path:
            self.output_file.set(path)
    
    def load_model(self):
        model_path = self.model_path.get()
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        
        try:
            self.status_var.set("Loading model... This may take a moment.")
            self.root.update()
            
            # Use default_config directly instead of calling load_config()
            config = default_config
            
            # Load tokenizers and models
            self.dna_tokenizer, self.text_tokenizer, dna_model, text_model = load_models(
                dna_model_path=config.model.dna_model_path,
                text_model_path=config.model.text_model_path,
                device=self.device
            )
            
            # Initialize model
            self.model = DNATextFusion(
                dna_model=dna_model,
                text_model=text_model,
                cross_attn_interval=config.model.cross_attn_interval
            ).to(self.device)
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Update status
            self.model.eval()
            self.model_info_var.set(f"Model loaded: {model_path}\nDevice: {self.device}")
            self.status_var.set("Model loaded successfully. Ready to generate.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model. Please check console for details.")
            print(f"Error loading model: {str(e)}")
    
    def generate_text_from_dna(self, dna_sequence, max_length=128):
        """Generate text description from a DNA sequence."""
        if self.model is None:
            return "Please load a model first."
        
        try:
            with torch.no_grad():
                # Tokenize DNA
                dna_inputs = self.dna_tokenizer(
                    dna_sequence, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                dna_inputs = {k: v.to(self.device) for k, v in dna_inputs.items()}
                
                # Generate text
                generated_ids = self.model.generate_text_from_dna(
                    dna_inputs, 
                    max_length=max_length
                )
                
                # Decode generated text
                generated_text = self.text_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                return generated_text
        except Exception as e:
            return f"Error generating text: {str(e)}"
    
    def generate_dna_from_text(self, text_description, max_length=512):
        """Generate DNA sequence from a text description."""
        if self.model is None:
            return "Please load a model first."
        
        try:
            with torch.no_grad():
                # Tokenize text
                text_inputs = self.text_tokenizer(
                    text_description, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                
                # Generate DNA
                generated_ids = self.model.generate_dna_from_text(
                    text_inputs, 
                    max_length=max_length
                )
                
                # Decode generated DNA
                generated_dna = self.dna_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                return generated_dna
        except Exception as e:
            return f"Error generating DNA: {str(e)}"
    
    def generate_single(self):
        """Handle single generation from the UI."""
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first.")
            return
        
        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showerror("Error", "Please enter input text.")
            return
        
        mode = self.mode.get()
        max_length = self.max_length.get()
        
        self.status_var.set(f"Generating {'text' if mode == 'dna2text' else 'DNA'}...")
        self.root.update()
        
        # Run generation in a separate thread to avoid freezing UI
        def run_generation():
            try:
                if mode == "dna2text":
                    result = self.generate_text_from_dna(input_text, max_length)
                else:
                    result = self.generate_dna_from_text(input_text, max_length)
                
                # Update UI with result (must use after method to run in main thread)
                self.root.after(0, lambda: self.update_output(result))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Generation failed: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set("Generation failed."))
        
        # Start generation in separate thread
        Thread(target=run_generation).start()
    
    def update_output(self, result):
        """Update the output text widget with generation result."""
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, result)
        self.status_var.set("Generation complete.")
    
    def copy_output(self):
        """Copy output text to clipboard."""
        output = self.output_text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(output)
        self.status_var.set("Output copied to clipboard.")
    
    def clear_output(self):
        """Clear the output text widget."""
        self.output_text.delete("1.0", tk.END)
    
    def save_output(self):
        """Save output text to a file."""
        output = self.output_text.get("1.0", tk.END).strip()
        if not output:
            messagebox.showerror("Error", "No output to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            title="Save Output As",
            filetypes=(("Text Files", "*.txt"), ("All files", "*.*")),
            defaultextension=".txt"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(output)
                self.status_var.set(f"Output saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def process_batch(self):
        """Process a batch of inputs from a CSV file."""
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first.")
            return
        
        input_file = self.input_file.get()
        if not os.path.exists(input_file):
            messagebox.showerror("Error", f"Input file not found: {input_file}")
            return
        
        output_file = self.output_file.get()
        mode = self.batch_mode.get()
        max_length = self.batch_max_length.get()
        
        try:
            # Load the CSV file
            df = pd.read_csv(input_file)
            
            # Check for required columns
            if mode == "dna2text" and "DNA_Sequence" not in df.columns:
                messagebox.showerror("Error", "Input CSV must have a 'DNA_Sequence' column for DNA to Text mode.")
                return
            
            if mode == "text2dna" and not any(col in df.columns for col in ["text", "Text"]):
                messagebox.showerror("Error", "Input CSV must have a 'text' or 'Text' column for Text to DNA mode.")
                return
            
            # Start batch processing in a separate thread
            self.progress["maximum"] = len(df)
            self.progress["value"] = 0
            self.progress_label.config(text=f"Processing 0/{len(df)} entries")
            self.progress_log.delete("1.0", tk.END)
            self.status_var.set(f"Processing batch... This may take a while.")
            
            def run_batch_processing():
                results = []
                
                try:
                    if mode == "dna2text":
                        # DNA to Text processing
                        for idx, row in df.iterrows():
                            dna_seq = row["DNA_Sequence"]
                            generated_text = self.generate_text_from_dna(dna_seq, max_length)
                            results.append({"DNA_Sequence": dna_seq, "Generated_Text": generated_text})
                            
                            # Update progress
                            self.root.after(0, lambda: self.update_batch_progress(idx, len(df), dna_seq, generated_text))
                    else:
                        # Text to DNA processing
                        text_col = "text" if "text" in df.columns else "Text"
                        for idx, row in df.iterrows():
                            text = row[text_col]
                            generated_dna = self.generate_dna_from_text(text, max_length)
                            results.append({text_col: text, "Generated_DNA": generated_dna})
                            
                            # Update progress
                            self.root.after(0, lambda: self.update_batch_progress(idx, len(df), text, generated_dna))
                    
                    # Save results
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(output_file, index=False)
                    
                    # Final update
                    self.root.after(0, lambda: self.progress_label.config(text=f"Completed processing {len(df)}/{len(df)} entries"))
                    self.root.after(0, lambda: self.status_var.set(f"Batch processing complete. Results saved to {output_file}"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"Batch processing complete. Results saved to {output_file}"))
                
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Batch processing failed: {str(e)}"))
                    self.root.after(0, lambda: self.status_var.set("Batch processing failed."))
            
            # Start batch processing thread
            Thread(target=run_batch_processing).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process batch: {str(e)}")
            self.status_var.set("Batch processing failed.")
    
    def update_batch_progress(self, idx, total, input_text, output_text):
        """Update batch processing progress in the UI."""
        self.progress["value"] = idx + 1
        self.progress_label.config(text=f"Processing {idx+1}/{total} entries")
        
        # Show abbreviated input/output in the log
        input_display = (input_text[:50] + "...") if len(input_text) > 50 else input_text
        output_display = (output_text[:50] + "...") if len(output_text) > 50 else output_text
        log_entry = f"Entry {idx+1}: {input_display} -> {output_display}\n"
        
        self.progress_log.insert(tk.END, log_entry)
        self.progress_log.see(tk.END)  # Scroll to the end

if __name__ == "__main__":
    root = tk.Tk()
    app = Helix2TextGUI(root)
    root.mainloop()