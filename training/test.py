# Create this as debug_imports.py in your AF-DETECTION root directory and run it

import os
import sys

print("=== IMPORT DEBUG ANALYSIS ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print()

# Check project structure
print("=== PROJECT STRUCTURE ===")
required_files = [
    'utils/__init__.py',
    'utils/data_processing.py', 
    'models/__init__.py',
    'models/spectral.py',
    'results/__init__.py',
    'results/model_results.py',
    'results/visualization.py',
    'training/train_spectral.py'
]

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path} - MISSING!")

print()

# Test individual imports
print("=== TESTING IMPORTS ===")

# Test 1: Can we import utils as a package?
try:
    import utils
    print("✅ 'import utils' successful")
    print(f"   utils location: {utils.__file__}")
    print(f"   utils.__all__: {getattr(utils, '__all__', 'Not defined')}")
except ImportError as e:
    print(f"❌ 'import utils' failed: {e}")

# Test 2: Can we import the data_processing module directly?
try:
    import utils.data_processing
    print("✅ 'import utils.data_processing' successful")
except ImportError as e:
    print(f"❌ 'import utils.data_processing' failed: {e}")

# Test 3: Can we import specific functions?
try:
    from utils.data_processing import load_and_segment_data
    print("✅ 'from utils.data_processing import load_and_segment_data' successful")
except ImportError as e:
    print(f"❌ 'from utils.data_processing import load_and_segment_data' failed: {e}")

# Test 4: Check if there are syntax errors in data_processing.py
try:
    with open('utils/data_processing.py', 'r') as f:
        content = f.read()
    
    # Try to compile the file to check for syntax errors
    compile(content, 'utils/data_processing.py', 'exec')
    print("✅ utils/data_processing.py syntax is valid")
except SyntaxError as e:
    print(f"❌ Syntax error in utils/data_processing.py: {e}")
except FileNotFoundError:
    print("❌ utils/data_processing.py not found")

# Test 5: Check models imports
try:
    from models.spectral import extract_spectral_features
    print("✅ 'from models.spectral import extract_spectral_features' successful")
except ImportError as e:
    print(f"❌ models.spectral import failed: {e}")

print()
print("=== PYTHON PATH ===")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

print()
print("Run this script from your AF-DETECTION root directory!")
print("If you see errors above, that's the issue to fix.")