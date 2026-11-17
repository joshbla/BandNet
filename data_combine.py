import numpy as np
import os
import time
import json


# Parameters

m =                                                 3
k =                                                 5
training_data1 =                                    1000000
training_data2 =                                    1500000
epochs1 =                                           5
epochs2 =                                           5
version1 =                                          "" # Can be empty
version2 =                                          "" # or could be (1) or (6)

epochs3 =                                           0
default_time =                                      3700
band_folder = f"{m}Band"


def main():

    # Calculate data in thousands and create folder paths
    data_thousands1 = int(training_data1 / 1000)
    folder1 = f"k-{k} training-{data_thousands1}k epochs-{epochs1}{version1}"
    path1 = os.path.join(band_folder, folder1, "data")

    data_thousands2 = int(training_data2 / 1000)
    folder2 = f"k-{k} training-{data_thousands2}k epochs-{epochs2}{version2}"
    path2 = os.path.join(band_folder, folder2, "data")

    # Calculate total data in thousands for the combined folder
    data_thousands3 = int((training_data1 + training_data2) / 1000)
    folder3 = f"k-{k} training-{data_thousands3}k epochs-{epochs3}"
    output_path = os.path.join(band_folder, folder3, "data")

    # Create the new folder
    counter = 1
    original_output_path = output_path
    while os.path.exists(output_path):
        output_path = f"{original_output_path}({counter})"
        counter += 1
    
    os.makedirs(output_path, exist_ok=True)

    # Read generation times from existing metadata
    generation_time1 = read_generation_time(folder1)
    generation_time2 = read_generation_time(folder2)

    # Calculate combined generation time
    combined_generation_time = generation_time1 + generation_time2

    # Create a new metadata.json in folder3 with combined generation time
    metadata = {"generation_time": combined_generation_time}
    with open(os.path.join(band_folder, folder3, "metadata.json"), 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    combine_npz_files(path1, path2, output_path)

def combine_npz_files(path1, path2, output_path):
    start = time.time()

    input_file1_path = os.path.join(path1, "inputs.npy")
    output_file1_path = os.path.join(path1, "outputs.npy")
    input_file2_path = os.path.join(path2, "inputs.npy")
    output_file2_path = os.path.join(path2, "outputs.npy")

    # Load data from both .npy files
    data1_input = np.load(input_file1_path)
    data1_output = np.load(output_file1_path)
    data2_input = np.load(input_file2_path)
    data2_output = np.load(output_file2_path)

    print("First dataset input shape:", data1_input.shape)
    print("Second dataset input shape:", data2_input.shape)
    print("First dataset output shape:", data1_output.shape)
    print("Second dataset output shape:", data2_output.shape)

    # Combine data
    combined_X = np.concatenate((data1_input, data2_input), axis=0)
    combined_y = np.concatenate((data1_output, data2_output), axis=0)

    print("Combined input shape:", combined_X.shape)
    print("Combined output shape:", combined_y.shape)

    end = time.time()
    #print the time it took to combine the data
    print(f"Data combined in {end - start} seconds")

    # Create output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    output_file_path_X = os.path.join(output_path, "inputs.npy")
    output_file_path_y = os.path.join(output_path, "outputs.npy")

    # Save combined data
    np.save(output_file_path_X, combined_X)
    np.save(output_file_path_y, combined_y)
    print(f"Combined data saved to {output_path}")

def read_generation_time(folder):
    metadata_path = os.path.join(band_folder, folder, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)
            return metadata.get('generation_time', default_time)  # Default if key not found
    return default_time

if __name__ == "__main__":
    # Run the main function for each version
    # for i in range(7):
    #     version1 = f"({2*i + 1})"
    #     version2 = f"({2*i + 2})"
    #     main()
    main()