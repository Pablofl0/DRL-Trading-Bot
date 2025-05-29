import os

def create_project_structure():
    """Create the necessary directories for the project."""
    
    # Create main directories
    directories = [
        'models',
        'results',
        'src/data',
        'src/env',
        'src/models',
        'src/utils'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("Project structure setup complete.")

if __name__ == "__main__":
    create_project_structure()