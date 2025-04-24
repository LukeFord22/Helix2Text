"""
Helix2Text Training Web Application

This script provides a web-based interface for training the Helix2Text model
without using the terminal. It allows configuring training parameters, selecting data files,
and monitoring training progress using Streamlit.

streamlit run /root/Helix2Text/helix2text_train_web.py
"""

import os
import sys
import yaml
import time
import socket
import threading
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from PIL import Image

# Set PyTorch memory management configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules directly from the MetaModel package
from MetaModel.configs.default_config import default_config
from MetaModel.scripts.train import train_model, load_config

# Set page configuration
st.set_page_config(
    page_title="Helix2Text - Model Training",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize all session state variables
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'stop_training' not in st.session_state:
    st.session_state.stop_training = False
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = {"loss": [], "bleu": []}
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'progress_value' not in st.session_state:
    st.session_state.progress_value = 0
if 'status_message' not in st.session_state:
    st.session_state.status_message = "Ready. Configure training parameters to begin."
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0
if 'current_loss' not in st.session_state:
    st.session_state.current_loss = 0.0
if 'current_bleu' not in st.session_state:
    st.session_state.current_bleu = 0.0
if 'current_lr' not in st.session_state:
    st.session_state.current_lr = 0.0
if 'current_eta' not in st.session_state:
    st.session_state.current_eta = 0.0
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if 'training_start_time' not in st.session_state:
    st.session_state.training_start_time = None
if 'dataset_info_text' not in st.session_state:
    st.session_state.dataset_info_text = "Dataset Information:\n\nName: Protein to DNA Sequence Dataset\nSource: Rsync Local Fetch\nFormat: CSV\n\nThis dataset contains protein descriptions and their corresponding DNA sequences, suitable for training the Helix2Text model to generate text descriptions from DNA sequences and vice versa."

# Initialize configuration-related session state variables
if 'config_path' not in st.session_state:
    st.session_state.config_path = ""
if 'train_data_path' not in st.session_state:
    st.session_state.train_data_path = "protein_to_dna_sequence_descriptions.csv"
if 'eval_data_path' not in st.session_state:
    st.session_state.eval_data_path = ""
if 'output_model_path' not in st.session_state:
    st.session_state.output_model_path = "best_model.pt"
if 'num_epochs' not in st.session_state:
    st.session_state.num_epochs = 10
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 2
if 'learning_rate' not in st.session_state:
    st.session_state.learning_rate = 5e-5
if 'early_stop_patience' not in st.session_state:
    st.session_state.early_stop_patience = 3
if 'grad_accum_steps' not in st.session_state:
    st.session_state.grad_accum_steps = 4
if 'use_wandb' not in st.session_state:
    st.session_state.use_wandb = True
if 'project_name' not in st.session_state:
    st.session_state.project_name = "Helix2Text"
if 'experiment_name' not in st.session_state:
    st.session_state.experiment_name = f"Training-{os.getpid()}"
if 'split_data' not in st.session_state:
    st.session_state.split_data = True
if 'train_data_fraction' not in st.session_state:
    st.session_state.train_data_fraction = 1.0
if 'train_split_fraction' not in st.session_state:
    st.session_state.train_split_fraction = 0.9
if 'download_path' not in st.session_state:
    st.session_state.download_path = os.getcwd()

# Thread-safe functions for session state access
def add_log_message(message):
    """Add a message to the log in a thread-safe way."""
    try:
        if 'log_messages' in st.session_state:
            st.session_state.log_messages.append(message)
    except:
        pass

def update_progress(value):
    """Update progress value in a thread-safe way."""
    try:
        st.session_state.progress_value = value
    except:
        pass

def update_current_epoch(value):
    """Update current epoch in a thread-safe way."""
    try:
        st.session_state.current_epoch = value
    except:
        pass

def update_current_loss(value):
    """Update current loss in a thread-safe way."""
    try:
        st.session_state.current_loss = value
    except:
        pass

def update_current_bleu(value):
    """Update current BLEU score in a thread-safe way."""
    try:
        st.session_state.current_bleu = value
    except:
        pass

def update_current_lr(value):
    """Update current learning rate in a thread-safe way."""
    try:
        st.session_state.current_lr = value
    except:
        pass

def update_current_eta(value):
    """Update current ETA in a thread-safe way."""
    try:
        st.session_state.current_eta = value
    except:
        pass

def update_status_message(message):
    """Update status message in a thread-safe way."""
    try:
        st.session_state.status_message = message
    except:
        pass

def set_is_training(value):
    """Set is_training status in a thread-safe way."""
    try:
        st.session_state.is_training = value
    except:
        pass

def get_stop_training_status():
    """Get stop_training status in a thread-safe way."""
    try:
        return st.session_state.stop_training
    except:
        return True  # Default to stopping if we can't access the state

def update_training_metrics(metrics):
    """Update training metrics in a thread-safe way."""
    try:
        st.session_state.training_metrics = metrics
    except:
        pass

def log(message):
    """Add a message to the log."""
    add_log_message(message)

# Main title
st.title("🧬 Helix2Text - Model Training")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Configuration", "Training", "Metrics", "Datasets"])

# Configuration Tab
with tab1:
    st.header("Configuration")
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Paths section
        st.subheader("Paths")
        
        config_path = st.text_input("Config File (Optional)", key="config_path")
        train_data_path = st.text_input("Training Data CSV", key="train_data_path")
        eval_data_path = st.text_input("Evaluation Data CSV", key="eval_data_path")
        output_model_path = st.text_input("Output Model Path", value="best_model.pt", key="output_model_path")
        
        # Training hyperparameters
        st.subheader("Training Hyperparameters")
        
        hyper_col1, hyper_col2 = st.columns(2)
        
        with hyper_col1:
            num_epochs = st.number_input("Number of Epochs", min_value=1, value=10, key="num_epochs")
            batch_size = st.number_input("Batch Size", min_value=1, value=8, key="batch_size")
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, value=5e-5, format="%.5f", key="learning_rate")
        
        with hyper_col2:
            early_stop_patience = st.number_input("Early Stop Patience", min_value=1, value=3, key="early_stop_patience")
            grad_accum_steps = st.number_input("Gradient Accumulation Steps", min_value=1, value=4, key="grad_accum_steps")
        
        # Weights & Biases logging
        st.subheader("Weights & Biases Logging")
        
        use_wandb = st.checkbox("Enable Weights & Biases Logging", value=True, key="use_wandb")
        
        if use_wandb:
            wandb_col1, wandb_col2 = st.columns(2)
            with wandb_col1:
                project_name = st.text_input("Project Name", value="Helix2Text", key="project_name")
            with wandb_col2:
                experiment_name = st.text_input("Experiment Name", value=f"Training-{os.getpid()}", key="experiment_name")
        
        # Data splitting options
        st.subheader("Data Splitting Options")
        
        split_data = st.checkbox("Enable Data Splitting", value=True, key="split_data")
        
        if split_data:
            split_col1, split_col2 = st.columns(2)
            with split_col1:
                train_data_fraction = st.number_input("Train Data Fraction", min_value=0.1, max_value=1.0, value=1.0, format="%.2f", key="train_data_fraction")
            with split_col2:
                train_split_fraction = st.number_input("Train Split Fraction", min_value=0.1, max_value=1.0, value=0.9, format="%.2f", key="train_split_fraction")
    
    with col2:
        # Config preview
        st.subheader("Configuration Preview")
        
        # Function to get config dictionary
        def get_config_dict():
            config = {
                "project_name": st.session_state.project_name if 'project_name' in st.session_state else "Helix2Text",
                "experiment_name": st.session_state.experiment_name if 'experiment_name' in st.session_state else f"Training-{os.getpid()}",
                "training": {
                    "num_epochs": st.session_state.num_epochs,
                    "batch_size": st.session_state.batch_size,
                    "learning_rate": st.session_state.learning_rate,
                    "early_stop_patience": st.session_state.early_stop_patience,
                    "grad_accum_steps": st.session_state.grad_accum_steps,
                    "use_wandb": st.session_state.use_wandb
                },
                "data": {
                    "train_file": st.session_state.train_data_path,
                    "eval_file": st.session_state.eval_data_path,
                    "split_data": st.session_state.split_data,
                    "train_data_fraction": st.session_state.train_data_fraction,
                    "train_split_fraction": st.session_state.train_split_fraction
                },
                "model": {
                    "output_path": st.session_state.output_model_path
                }
            }
            return config
        
        # Function to generate YAML preview
        def update_config_preview():
            config = get_config_dict()
            
            preview_text = "# Helix2Text Training Configuration\n\n"
            
            for section, values in config.items():
                preview_text += f"{section}:\n"
                if isinstance(values, dict):
                    for key, value in values.items():
                        preview_text += f"  {key}: {value}\n"
                else:
                    preview_text += f"  {values}\n"
                preview_text += "\n"
            
            return preview_text
        
        # Display preview
        config_preview = st.text_area("Configuration YAML", value=update_config_preview(), height=400, key="config_preview")
        
        # Config file operations
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            # Function to load config
            def load_config_from_file():
                if not st.session_state.config_path:
                    st.warning("No config file selected.")
                    return
                
                try:
                    config = load_config(st.session_state.config_path)
                    
                    # Update session state with loaded config
                    st.session_state.num_epochs = config.training.num_epochs
                    st.session_state.batch_size = config.training.batch_size
                    st.session_state.learning_rate = config.training.learning_rate
                    st.session_state.early_stop_patience = config.training.early_stop_patience
                    st.session_state.grad_accum_steps = config.training.grad_accum_steps
                    
                    if hasattr(config, "project_name"):
                        st.session_state.project_name = config.project_name
                    if hasattr(config, "experiment_name"):
                        st.session_state.experiment_name = config.experiment_name
                    
                    if hasattr(config.data, "train_file"):
                        st.session_state.train_data_path = config.data.train_file
                    if hasattr(config.data, "eval_file"):
                        st.session_state.eval_data_path = config.data.eval_file
                    if hasattr(config.data, "split_data"):
                        st.session_state.split_data = config.data.split_data
                    if hasattr(config.data, "train_data_fraction"):
                        st.session_state.train_data_fraction = config.data.train_data_fraction
                    if hasattr(config.data, "train_split_fraction"):
                        st.session_state.train_split_fraction = config.data.train_split_fraction
                    
                    st.success(f"Config loaded from {st.session_state.config_path}")
                    
                    # Force rerun to update UI
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Failed to load config: {str(e)}")
            
            if st.button("Load Config"):
                load_config_from_file()
        
        with config_col2:
            # Function to save config
            def save_config_to_file():
                config_dict = get_config_dict()
                
                # Create a download button
                yaml_str = yaml.dump(config_dict, default_flow_style=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download Config YAML",
                    data=yaml_str,
                    file_name=f"helix2text_config_{timestamp}.yaml",
                    mime="text/yaml"
                )
            
            save_config_to_file()  # This creates the download button

# Training Tab
with tab2:
    st.header("Training Control")
    
    # Training controls
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        # Function to start training
        def start_training():
            if st.session_state.is_training:
                st.warning("Training is already in progress.")
                return
            
            # Validate inputs
            if not validate_training_inputs():
                return
            
            # Update state
            set_is_training(True)
            st.session_state.stop_training = False
            update_progress(0)
            st.session_state.log_messages = []
            update_status_message("Training started...")
            st.session_state.training_start_time = time.time()
            
            # Reset metrics tracking
            update_training_metrics({"loss": [], "bleu": []})
            
            # Get configuration
            config = get_config_object()
            
            # Start training in a separate thread
            training_thread = threading.Thread(target=run_training, args=(config,))
            training_thread.daemon = True
            training_thread.start()
        
        start_button = st.button("Start Training", disabled=st.session_state.is_training, on_click=start_training)
    
    with control_col2:
        # Function to stop training
        def stop_training_process():
            if not st.session_state.is_training:
                return
                
            st.session_state.stop_training = True
            update_status_message("Stopping training after current epoch...")
            log("\n[INFO] Training stop requested. Will stop after current epoch completes.")
        
        stop_button = st.button("Stop Training", disabled=not st.session_state.is_training, on_click=stop_training_process)
    
    # Training progress
    st.subheader("Training Progress")
    
    progress_bar = st.progress(st.session_state.progress_value / 100.0)
    st.text(f"Progress: {st.session_state.progress_value:.1f}%")
    
    # Live metrics
    st.subheader("Current Training Metrics")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Epoch", f"{st.session_state.current_epoch}")
        st.metric("Loss", f"{st.session_state.current_loss:.4f}")
    
    with metric_col2:
        st.metric("BLEU", f"{st.session_state.current_bleu:.4f}")
        st.metric("Learning Rate", f"{st.session_state.current_lr:.6f}")
    
    with metric_col3:
        elapsed_time = 0
        if st.session_state.training_start_time:
            elapsed_time = time.time() - st.session_state.training_start_time
        
        st.metric("Elapsed Time", f"{elapsed_time/60:.1f} min")
        st.metric("Status", st.session_state.status_message)
    
    # Training log
    st.subheader("Training Log")
    
    log_col1, log_col2 = st.columns([4, 1])
    
    with log_col1:
        log_container = st.container()
        with log_container:
            if st.session_state.log_messages:
                st.text_area("Log Output", 
                           value="\n".join(st.session_state.log_messages),
                           height=400,
                           key="log_display")
            else:
                st.text_area("Log Output", value="No logs yet. Start training to see logs.", height=400, key="log_display")
    
    with log_col2:
        # Function to clear log
        def clear_log():
            st.session_state.log_messages = []
        
        st.button("Clear Log", on_click=clear_log)
        
        # Function to save log
        def save_log():
            if not st.session_state.log_messages:
                st.warning("No log messages to save.")
                return
            
            log_text = "\n".join(st.session_state.log_messages)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="Download Training Log",
                data=log_text,
                file_name=f"helix2text_training_log_{timestamp}.txt",
                mime="text/plain"
            )
        
        if st.session_state.log_messages:
            save_log()  # This creates the download button

# Metrics Tab  
with tab3:
    st.header("Training Metrics Visualization")
    
    # Create plots if there's data
    if st.session_state.training_metrics["loss"]:
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            # Loss plot
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            epochs = list(range(1, len(st.session_state.training_metrics["loss"])+1))
            ax1.plot(epochs, st.session_state.training_metrics["loss"], 'b-', marker='o')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            st.pyplot(fig1)
        
        with metrics_col2:
            # BLEU score plot
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            epochs = list(range(1, len(st.session_state.training_metrics["bleu"])+1))
            ax2.plot(epochs, st.session_state.training_metrics["bleu"], 'g-', marker='o')
            ax2.set_title('BLEU Score')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('BLEU')
            ax2.grid(True)
            st.pyplot(fig2)
        
        # Export metrics as CSV
        def export_metrics_csv():
            epochs = list(range(1, len(st.session_state.training_metrics["loss"])+1))
            
            df = pd.DataFrame({
                "epoch": epochs,
                "loss": st.session_state.training_metrics["loss"],
                "bleu": st.session_state.training_metrics["bleu"]
            })
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Metrics CSV",
                data=csv,
                file_name=f"helix2text_metrics_{timestamp}.csv",
                mime="text/csv"
            )
        
        # Export charts
        def save_charts():
            # Generate a figure with both plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            epochs = list(range(1, len(st.session_state.training_metrics["loss"])+1))
            
            # Plot loss
            ax1.plot(epochs, st.session_state.training_metrics["loss"], 'b-', marker='o')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            
            # Plot BLEU
            ax2.plot(epochs, st.session_state.training_metrics["bleu"], 'g-', marker='o')
            ax2.set_title('BLEU Score')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('BLEU')
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Convert to image
            fig.canvas.draw()
            img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            
            # Convert to bytes
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Charts as PNG",
                data=img_bytes,
                file_name=f"helix2text_charts_{timestamp}.png",
                mime="image/png"
            )
        
        metrics_export_col1, metrics_export_col2 = st.columns(2)
        
        with metrics_export_col1:
            export_metrics_csv()  # This creates the download button
            
        with metrics_export_col2:
            save_charts()  # This creates the download button
    else:
        st.info("No training metrics available. Start a training session to visualize metrics.")

# Datasets Tab
with tab4:
    st.header("Dataset Management")
    
    # Available datasets section
    st.subheader("Available Datasets")
    
    dataset_frame = st.container()
    with dataset_frame:
        st.markdown("### Protein to DNA Sequence Descriptions")
        st.markdown("A curated dataset of protein descriptions mapped to their corresponding DNA sequences.")
        
        # Download location
        download_path = st.text_input("Download Location", value=os.getcwd(), key="download_path")
        
        # Server details for rsync command
        server_username = os.environ.get('USER', os.environ.get('USERNAME', 'username'))
        
        # Get hostname
        def get_hostname():
            try:
                return socket.gethostname()
            except:
                return "server"
        
        server_hostname = get_hostname()
        
        # Transfer dataset dialog
        if st.button("Transfer Dataset"):
            transfer_container = st.container()
            with transfer_container:
                st.markdown(r"""
                Follow these steps to transfer your dataset from your local PC to this server:
                
                Path: \Users\lhfor\AppData\Local\Programs\Python\Python38-32\Helix2TextDataRetreival\Dataset\protein_to_dna_sequence_descriptions.csv
                
                1. This feature generates an SCP command for you to run on your local PC.
                2. Copy the generated command and run it in your local terminal/command prompt.
                3. It will securely transfer your dataset to this server with progress tracking.
                4. After transferring is complete, check for the file and register it for training.
                
                Note: SCP must be installed on your local PC:
                - Mac/Linux: Included by default
                - Windows: Use PowerShell's pscp or install an SCP client.
                """)
                
                # Local file path input
                local_path = st.text_input("Path to dataset on your local PC:", 
                                        placeholder="/home/user/datasets/protein_to_dna.csv or C:\\Data\\dataset.csv")
                
                # Generate SCP command
                if local_path:
                    # Adjust path format for Windows vs Unix
                    clean_path = local_path.replace('\\', '/')
                    
                    # Build the transfer command using SCP instead of rsync
                    scp_cmd = f"scp \"{clean_path}\" {server_username}@{server_hostname}:{download_path}"
                    
                    st.code(scp_cmd, language="bash")
                    
                    # Add alternative commands for different environments
                    st.markdown("#### Alternative Transfer Methods")
                    st.markdown("If SCP doesn't work, try one of these alternatives:")
                    
                    # SFTP alternative
                    sftp_cmd = f"""sftp {server_username}@{server_hostname}
cd {download_path}
put "{clean_path}"
exit"""
                    st.code(sftp_cmd, language="bash")
                    
                    # PowerShell alternative for Windows users
                    ps_cmd = f"pscp \"{local_path}\" {server_username}@{server_hostname}:{download_path}"
                    st.code(ps_cmd, language="powershell")
                    
                    # Copy to clipboard button - note: uses JavaScript so only works in browser
                    st.markdown(f"""
                    <button onclick="navigator.clipboard.writeText('{scp_cmd}')">
                        Copy SCP command to clipboard
                    </button>
                    """, unsafe_allow_html=True)
                
                # Check file transfer
                st.subheader("Check File Transfer")
                
                def check_file_exists():
                    if not local_path:
                        st.warning("Please enter the path to your dataset file.")
                        return
                        
                    # Extract filename from path
                    filename = os.path.basename(local_path.replace('\\', '/'))
                    target_path = os.path.join(download_path, filename)
                    
                    if os.path.exists(target_path):
                        # File exists, offer to use it
                        try:
                            df = pd.read_csv(target_path)
                            row_count = len(df)
                            col_count = len(df.columns)
                            
                            st.success(f"File found: {filename} ({row_count} rows, {col_count} columns)")
                            
                            # Update dataset info
                            update_dataset_info(df, target_path)
                            
                            # Set as training data path if it's empty
                            if not st.session_state.train_data_path:
                                st.session_state.train_data_path = target_path
                                st.success("Dataset path has been set as the training data path.")
                            
                            return True
                        except Exception as e:
                            st.error(f"File found but appears to be invalid: {str(e)}")
                            return False
                    else:
                        st.warning(f"The file {filename} has not been transferred yet. Please run the SCP command on your local PC first.")
                        return False
                
                if st.button("Check for File"):
                    check_file_exists()
        
        # Dataset information section
        st.subheader("Dataset Information")
        
        st.text_area("Dataset Details", value=st.session_state.dataset_info_text, height=200)

# Helper functions
def update_dataset_info(df, file_path):
    """Update the dataset information text with detailed information."""
    # Calculate file size
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
    
    # Update session state
    st.session_state.dataset_info_text = info_text

def validate_training_inputs():
    """Validate that all required inputs are provided."""
    # Check for training data
    if not st.session_state.train_data_path:
        st.error("Training data file is required.")
        return False
    
    if not os.path.exists(st.session_state.train_data_path):
        st.error(f"Training data file not found: {st.session_state.train_data_path}")
        return False
    
    # Check for eval data
    if st.session_state.eval_data_path and not os.path.exists(st.session_state.eval_data_path):
        st.error(f"Evaluation data file not found: {st.session_state.eval_data_path}")
        return False
    
    # Check hyperparameters
    try:
        if st.session_state.num_epochs <= 0:
            st.error("Number of epochs must be positive.")
            return False
            
        if st.session_state.batch_size <= 0:
            st.error("Batch size must be positive.")
            return False
            
        if st.session_state.learning_rate <= 0:
            st.error("Learning rate must be positive.")
            return False
    
    except Exception:
        st.error("Invalid numeric values in hyperparameters.")
        return False
        
    return True

def get_config_object():
    """Create a configuration object that matches the expected format."""
    # Start with default configuration
    config = default_config
    
    # Update with GUI values
    config.project_name = st.session_state.project_name if 'project_name' in st.session_state else "Helix2Text"
    config.experiment_name = st.session_state.experiment_name if 'experiment_name' in st.session_state else f"Training-{os.getpid()}"
    
    # Update training configuration
    config.training.num_epochs = st.session_state.num_epochs
    config.training.batch_size = st.session_state.batch_size
    config.training.learning_rate = st.session_state.learning_rate
    config.training.early_stop_patience = st.session_state.early_stop_patience
    config.training.grad_accum_steps = st.session_state.grad_accum_steps
    
    # Update data configuration
    if st.session_state.train_data_path:
        config.data.train_file = st.session_state.train_data_path
    if st.session_state.eval_data_path:
        config.data.eval_file = st.session_state.eval_data_path
    config.data.split_data = st.session_state.split_data
    config.data.train_data_fraction = st.session_state.train_data_fraction
    config.data.train_split_fraction = st.session_state.train_split_fraction
    
    # Update model output path
    config.model.output_path = st.session_state.output_model_path
    
    return config

def run_training(config):
    """Execute the training process."""
    try:
        import traceback
        import sys
        
        # Print detailed information to console
        print("\n==== TRAINING STARTED ====")
        print(f"Training data path: {config.data.train_file}")
        print(f"Output model path: {config.model.output_path}")
        print(f"Batch size: {config.training.batch_size}")
        print(f"Epochs: {config.training.num_epochs}")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
        # Check if training file exists
        if not os.path.exists(config.data.train_file):
            error_msg = f"Training data file not found: {config.data.train_file}"
            print(error_msg)
            add_log_message(error_msg)
            update_status_message(f"Training failed: {error_msg}")
            return
            
        # Print GPU information
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        
        # Create custom handlers to intercept training progress
        class LogCallback:
            def __init__(self):
                self.epoch = 0
                self.total_epochs = config.training.num_epochs
                # Keep local copies of metrics
                self.training_metrics = {"loss": [], "bleu": []}
            
            def on_epoch_begin(self, epoch):
                self.epoch = epoch
                add_log_message(f"\n[Epoch {epoch+1}/{self.total_epochs}] Starting...\n")
                update_progress((epoch / self.total_epochs) * 100)
                update_current_epoch(epoch + 1)
            
            def on_epoch_end(self, epoch, metrics):
                # Update metrics
                avg_loss = metrics.get("avg_loss", 0)
                bleu = metrics.get("bleu_score", 0)
                
                # Add to local metrics history
                self.training_metrics["loss"].append(avg_loss)
                self.training_metrics["bleu"].append(bleu)
                
                # Update the global metrics in a thread-safe way
                update_training_metrics(self.training_metrics)
                
                # Log epoch results
                add_log_message(f"[Epoch {epoch+1}/{self.total_epochs}] Loss: {avg_loss:.4f} | BLEU: {bleu:.4f}")
                
                # Update progress and metrics in a thread-safe way
                update_progress(((epoch + 1) / self.total_epochs) * 100)
                update_current_loss(avg_loss)
                update_current_bleu(bleu)
                
                # Check if we should stop - use a thread-safe way to access this
                return get_stop_training_status()
            
            def on_step_end(self, step, step_metrics):
                if step % 10 == 0:  # Log only every 10 steps to avoid flooding
                    loss = step_metrics.get("step_loss", 0)
                    lr = step_metrics.get("learning_rate", 0)
                    eta = step_metrics.get("eta_min", None)
                    
                    # Update metrics in a thread-safe way
                    update_current_loss(loss)
                    update_current_lr(lr)
                    if eta is not None:
                        update_current_eta(eta)
        
        # Create a callback instance
        callback = LogCallback()
        
        # Disable wandb if not requested - use thread-safe check
        use_wandb = False
        try:
            use_wandb = st.session_state.use_wandb
        except:
            pass
            
        if not use_wandb:
            # Set the environment variable to use a dummy wandb implementation
            os.environ["WANDB_MODE"] = "disabled"
        
        # Add our custom callback to config
        config.callbacks = {"training": callback}
        
        add_log_message("Initializing training...")
        add_log_message(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
        # Start training with the callback
        model, metrics = train_model(config)
        
        # Log final results
        add_log_message("\n============= Training Complete =============")
        add_log_message(f"Final BLEU Score: {metrics['bleu_score']:.4f}")
        add_log_message(f"Model saved to: {config.model.output_path}")
        
        # Update status message in a thread-safe way
        update_status_message("Training completed successfully!")
    
    except Exception as e:
        print("\n==== TRAINING ERROR ====")
        print(f"Error: {str(e)}")
        traceback.print_exc(file=sys.stdout)  # Print full stack trace
        add_log_message(f"ERROR: {str(e)}")
        update_status_message(f"Training failed: {str(e)}")
    
    finally:
        print("\n==== TRAINING FINISHED ====")
        set_is_training(False)