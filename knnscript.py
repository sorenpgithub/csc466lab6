import subprocess
import sys

def run_script(script_name, arguments):
    # Construct the command to run the script
    command = [sys.executable, script_name] + arguments

    try:
        # Run the command and wait for the process to finish
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("started")
    # List of script names to run
    script_arguments = ["vectors/vectors_stem_stop.csv", "vectors/vectors_stop.csv", "vectors/vectors_stem.csv", "vectors/vectors.csv"]
    for path in script_arguments:
        # Specify the arguments to pass to each script
        

        # Run the script
        print(path)
        run_script("knnAuthorship.py", [path, "ground_truth.csv", "10", "-o", "-m"])

    print("All scripts have been executed.")