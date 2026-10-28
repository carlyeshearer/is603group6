## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites
Make sure you have **Python 3** installed.

### 1. Clone the Repository
Open your terminal, go to your desired directory, and run:
```bash
git clone https://github.com/carlyeshearer/is603group6.git
cd is603group6
```

### 2. Create and Activate Virtual Environment
Create a virtual environment to manage dependencies:

#### On Windows PowerShell:
```powershell
#Create environment
python -m venv .venv

#Activate environment
.\.venv\Scripts\Activate.ps1
```

#### On Linux / macOS:
```bash
#Create environment
python3 -m venv .venv

#Activate environment
source .venv/bin/activate
```

### 3. Install Dependencies
Once the virtual environment is active, install the required packages:
```bash
pip install -r requirements.txt
```

### 4. Run the Code
Execute the application files:

```bash
#Churn analysis
python churn.py

#Customer insights
python customer.py

#Product script (To be implemented)
python product.py
```