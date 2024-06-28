# Audio Deepfake Detection

### Development Environment Setup

To set up the development environment for this project, follow these steps:

1. **Run the setup script**:

   This script `setup_conda_env.sh` will create a new conda environment, set the `PYTHONPATH`, and install the required dependencies.

   ```sh
   cd AudioDeepfakeDetection
   chmod +x setup_conda_env.sh
   ./setup_conda_env.sh
   ```

2. **Activate the environment**:

   Once the setup is complete, activate the conda environment:

   ```sh
   conda activate audio_deepfake_env
   ```

3. **Verify the `PYTHONPATH`**:

   You can verify that the `PYTHONPATH` is set correctly by running:

   ```sh
   echo $PYTHONPATH
   ```

4. **Open Jupyter lab or notebook**:

   Start Jupyter Lab or Notebook:

   ```sh
   jupyter lab
   ```
   or

   ```sh
   jupyter notebook
   ```

**Note**

- **Dependencies**: All dependencies are listed in the `requirements.txt` file and will be installed automatically by the setup script.
- **Custom Python Path**: The `PYTHONPATH` is set to include the `src` directory of the project, allowing for easy imports of all modules within the codebase.
