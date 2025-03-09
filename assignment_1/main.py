import subprocess

def main():
    # Step 1: Create dataset splits
    print("Creating dataset splits...")
    subprocess.run(["python", "split_data.py"], check=True)
    
    # Step 2: Run the hyperparameter sweep
    print("\nStarting hyperparameter sweep...")
    subprocess.run(["python", "sweep.py"], check=True)

if __name__ == "__main__":
    main()
