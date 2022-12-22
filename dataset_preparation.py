import os
import zipfile


def extract_dataset(zipped_dataset, output_directory):
    """
        Extracts the dataset from the zipped file and moves the folders to the root directory.
        :param zipped_dataset: The path to the zipped dataset.
        :param output_directory: The path to the output directory.
        :return: None
    """
    os.makedirs(output_directory, exist_ok=True)
    with zipfile.ZipFile(zipped_dataset, 'r') as zip_ref:
        zip_ref.extractall(output_directory)


if __name__ == '__main__':
    extract_dataset('training_dataset_homework2.zip', 'dataset/')
