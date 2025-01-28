import os
import shutil

def delete_data_subfolders(directory):
    # Iterate over all the items in the specified directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Check if the item is a directory
        if os.path.isdir(item_path):
            data_subfolder_path = os.path.join(item_path, 'data')
            
            # Check if the data subfolder exists in this directory
            if os.path.exists(data_subfolder_path) and os.path.isdir(data_subfolder_path):
                try:
                    # Remove the data subfolder
                    shutil.rmtree(data_subfolder_path)
                    print(f"Deleted '{data_subfolder_path}' successfully.")
                except Exception as e:
                    print(f"Error deleting '{data_subfolder_path}': {e}")
            else:
                print(f"No 'data' subfolder in '{item_path}'.")

if __name__ == "__main__":
    # 1Band, 2Band, and 3Band directories
    for i in range(1, 4):
        models_directory = f'{i}Band'
        if os.path.exists(models_directory):
            delete_data_subfolders(models_directory)
        else:
            print(f"Directory '{models_directory}' does not exist.")