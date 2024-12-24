
import sys
import os

# Get the absolute path to the 'src' directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Add 'src' directory to the Python path if it's not already present
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Optional: Print the updated sys.path for debugging purposes
print("Updated sys.path:", sys.path)

# Now you can import modules from the 'src' directory in your test scripts
