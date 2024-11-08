import dill 
import os 
import sys





# Function to save a Python object to a file
def save_object(file_path, obj):
    try:
        # Get the directory path from the file path
        dir_path = os.path.dirname(file_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Open the file in write-binary mode and save the object using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception:
        # Raise a custom exception if any error occurs
        raise ValueError('Object save failed')


# Function to load a Python object from a file
def load_object(file_path):
    try:
        # Open the file in read-binary mode and load the object using dill
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        # Raise a custom exception if any error occurs
        raise ValueError('Object access failed ')
