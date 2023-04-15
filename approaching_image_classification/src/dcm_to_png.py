# Copied from chatgpt
import os
import pydicom
from PIL import Image


# Define input and output directories
input_dir = '/Users/insomni_.ak/Documents/Machine Learning/AAAMLP/AAAMLP/approaching_image_classification/input/siim_png/train_dcm'
output_dir = '/Users/insomni_.ak/Documents/Machine Learning/AAAMLP/AAAMLP/approaching_image_classification/input/siim_png/train_png'

print("start")
# Loop through each DCM file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.dcm'):
        # Read the DCM file using PyDICOM
        ds = pydicom.dcmread(os.path.join(input_dir, filename))
        
        # Convert the pixel data to a Pillow Image object
        image = Image.fromarray(ds.pixel_array)
        
        # Save the image as a PNG file in the output directory
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)
        image.save(output_path)
print("Done!")