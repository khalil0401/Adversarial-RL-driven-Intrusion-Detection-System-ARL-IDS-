# Setup Guide for ARL-IDS

This guide will help you set up the Adversarial-RL-driven Intrusion Detection System (ARL-IDS) on your machine.

## Prerequisites

- **OS**: Windows, Linux, or macOS
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU recommended for faster training (CUDA compatible)

## 1. Installation

### Option A: Manual Installation

1.  **Clone or Download the Repository**
    Ensure you are in the project root directory.

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 2. Dataset Configuration

The system requires the **ToN_IoT** dataset (specifically `train_test_network.csv`).

1.  **Download the Dataset**
    Download `Train_Test_Network.csv` from the official [ToN_IoT repository](https://research.unsw.edu.au/projects/toniot-datasets).

2.  **Configure the Path**
    You have two options to tell the system where the file is:

    **Option 1: Environment Variable (Recommended)**
    Set the `ARL_IDS_DATA_PATH` environment variable.
    
    *Windows (PowerShell)*:
    ```powershell
    $env:ARL_IDS_DATA_PATH = "C:\path\to\your\Train_Test_Network.csv"
    ```
    
    *Windows (CMD)*:
    ```cmd
    set ARL_IDS_DATA_PATH=C:\path\to\your\Train_Test_Network.csv
    ```

    **Option 2: Command Line Argument**
    Pass the path directly when running scripts:
    ```bash
    python src/run_Training.py --data_path "C:\path\to\Train_Test_Network.csv"
    ```

    **Option 3: Default Location**
    Place the file at `src/train_test_network.csv` inside the project folder.

---

## 3. Running the System

### Train the Model
This will train the Autoencoder (Layer 1) and the RL Agents (Layer 2 & 3).

```bash
# Basic run
python src/run_Training.py

# With custom parameters
python src/run_Training.py --episodes 100000 --data_path "path/to/data.csv"
```

### Evaluate the Model
After training, evaluate the performance against the adversary.

```bash
python src/evaluate.py --data_path "path/to/data.csv"
```

### Run Ablation Studies
Run experiments to verify the contribution of each component.

```bash
python src/ablation.py
```

---

## Troubleshooting

**Q: `FileNotFoundError: Dataset not found`**
A: Double-check your path. Use the environment variable method to be sure.
   print out `config.py` to see what path it is looking for: `python src/config.py`

**Q: `ImportError: No module named src`**
A: Make sure you are running the command from the **root** folder of the project, not inside `src`.
   CORRECT: `python src/run_Training.py`
   INCORRECT: `cd src` then `python run_Training.py`

**Q: CUDA out of memory**
A: The batch size might be too large for your GPU. Try forcing CPU mode:
   (Windows) `$env:FORCE_CPU="1"; python src/run_Training.py`
