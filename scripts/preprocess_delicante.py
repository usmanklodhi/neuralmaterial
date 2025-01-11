from PIL import Image
import os


def process_images(input_folder, output_folder, num_files):
    # Ensure the output folder exists
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    # Sort files for consistent processing order
    image_files.sort()

    # Only pick the first `num_files` images
    if len(image_files) > num_files:
        image_files = image_files[:num_files]

    total_files = len(image_files)

    # Determine the split point for train (80%) and test (20%)
    train_count = round(0.8 * total_files)
    test_count = total_files - train_count

    # Process images and split into train and test folders
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"File not found: {image_path}")
            continue

        # Define the width of each individual map (assuming equal division horizontally)
        image_width, image_height = image.size
        individual_width = image_width // 5  # 5 maps horizontally

        # Define the filenames for the individual maps
        map_names = ['input', 'normal', 'diffuse', 'roughness', 'specular']

        # Determine whether this image goes into the train or test set
        if idx < train_count:
            set_folder = train_folder
        else:
            set_folder = test_folder

        # Create a numbered subdirectory for the file
        subfolder_name = f"{idx + 1:04d}"  # Zero-padded numbering (e.g., 0001, 0002)
        subfolder_path = os.path.join(set_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # Subdirectory for BRDF maps
        brdf_folder = os.path.join(subfolder_path, 'brdf')
        os.makedirs(brdf_folder, exist_ok=True)

        # Process and save the maps
        for map_idx, map_name in enumerate(map_names):
            # Crop the image for the current map
            map_img = image.crop((map_idx * individual_width, 0, (map_idx + 1) * individual_width, image_height))
            if map_name == 'input':
                output_path = os.path.join(subfolder_path, f'{map_name}.png')
            else:
                output_path = os.path.join(brdf_folder, f'{map_name}.png')

            map_img.save(output_path)

        print(f"Processed and saved maps for: {image_file} -> {set_folder}/{subfolder_name}")


def main():
    # Define the input and output folders
    input_folder = '../train_blended_limited'  # Replace with your input folder path
    output_folder = '../trainingDelicante'  # Replace with your desired output folder path

    # Number of files to process
    num_files = 400  # Specify the number of files you want to process

    # Process the images
    process_images(input_folder, output_folder, num_files)


if __name__ == "__main__":
    main()
