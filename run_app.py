import subprocess
import webbrowser
import time
import os

# Function to install dependencies
def install_dependencies():
    try:
        print("Installing dependencies from requirements.txt...")
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing dependencies: {e}")
        exit(1)

# Function to run the training pipeline
def run_training_pipeline():
    try:
        print("Running training pipeline...")
        # Run the train_pipeline.py
        subprocess.run(["python", "src/pipeline/train_pipeline.py"], check=True)
        print("Training pipeline completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during training pipeline execution: {e}")
        exit(1)

# Function to run the Flask app
def run_flask_app():
    try:
        print("Starting the Flask app...")
        # Start the Flask app in a separate process
        subprocess.Popen(["python", "app.py"])
        
        # Wait a few seconds to ensure Flask starts
        time.sleep(3)
        
        # Open the default web browser
        webbrowser.open("http://localhost:5000", new=2)  # Open in a new tab, if possible
        print("Flask app is running. Opened browser at http://localhost:5000")
    except Exception as e:
        print(f"Error during Flask app execution: {e}")
        exit(1)

if __name__ == "__main__":
    # Step 1: Install dependencies
    install_dependencies()

    # Step 2: Run the training pipeline
    run_training_pipeline()
    
    # Step 3: Run the Flask app and open the browser
    run_flask_app()
