# +
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from random import choice

def binary_mask(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Display the grayscale image using matplotlib
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()
    
    # Apply a threshold to create a binary mask
    # The threshold value may need to be djusted for different images
    threshold_value = 128
    _, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Display the grayscale image using matplotlib
    plt.imshow(cleaned_mask, cmap='gray')
    plt.title('Binary Mask')
    plt.axis('off')
    plt.show()
    
    return cleaned_mask

imgs_list = range(1, 398)

for i in range(3):
    idx = choice(imgs_list)
    
    path = f"/storage/ice1/0/3/sprathipati6/fml/proj/hemanth_output_imgs/output_img_{idx}.png"
    
    _ = binary_mask(path)

# To save the binary mask:
# cv2.imwrite('path_to_save_binary_mask.png', binary_mask)


# +
import cv2
import numpy as np

# Load the image
image_path = path
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Canny edge detection
edges = cv2.Canny(gray, threshold1=30, threshold2=100)

# Dilate the edges to close the gaps
dilation_kernel = np.ones((5,5), np.uint8)
dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=2)

# Find contours and fill them
contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(gray)  # Create a mask where white is what we want, black otherwise
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# Save or process the 'mask' as needed
# Display the grayscale image using matplotlib
plt.imshow(mask, cmap='gray')
plt.title('Binary Mask')
plt.axis('off')
plt.show()
# -

# !module load anaconda3

# +
##################################
# Find the best input SAM to give
# #################################

import numpy as np
import os

# Set the directory where the files are located
directory = "/home/hice1/sprathipati6/scratch/fml/proj/Prompting results/Bus/st1/scores"

# Initialize variables to keep track of the maximum score and corresponding file
max_score = -np.inf  # Start with negative infinity
max_score_file = ""

# Iterate through the range of files
for i in range(401):  # Assuming the files are named from 0score.npy to 400score.npy
    file_name = f"{i}score.npy"
    file_path = os.path.join(directory, file_name)
    
    # Ensure the file exists before trying to load it
    if os.path.isfile(file_path):
        # Load the array from the file
        scores_array = np.load(file_path)

        
        if len(scores_array > 0):
            # Get the last element in the array, which is the final score
            final_score = scores_array[-1]

            # Update the maximum score and corresponding file if this score is greater
            if final_score > max_score:
                max_score = final_score
                max_score_file = file_name
        else:
            print(f"Skipping empty for {file_path}")

# Print out the results
print(f"The file with the maximum final score is: {max_score_file}, with a score of {max_score}")


# +
##################################
# Look at the contents of files
# #################################

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from random import choice
import numpy as np

def show_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

samples = np.load('/storage/ice1/0/3/sprathipati6/fml/proj/samples.npy', allow_pickle=True)
labels = np.load('/storage/ice1/0/3/sprathipati6/fml/proj/labels.npy', allow_pickle=True)

print(samples.size)

IDX = 95

scores = np.load(f'/home/hice1/sprathipati6/scratch/fml/proj/Prompting results/Bus/st1/scores/{IDX}score.npy')
mask_file = f'/storage/ice1/0/3/sprathipati6/fml/proj/Prompting results/Bus/st1/masks/{IDX}_0_mask.png'

green_pts_file = "/storage/ice1/0/3/sprathipati6/fml/proj/Prompting results/Bus/st1/points/95_green.npy"
red_pts_file = "/storage/ice1/0/3/sprathipati6/fml/proj/Prompting results/Bus/st1/points/95_red.npy"

green = np.load(green_pts_file, allow_pickle=True)
red = np.load(red_pts_file, allow_pickle=True)

print(green)
print(red)

show_image(samples[IDX])
show_image(labels[IDX])

check_mask = False
if check_mask:
    unique_values, counts = np.unique(labels[0], return_counts=True)
    print(dict(zip(unique_values, counts)))
    
# Load this supposedly best mask
mask_img = np.array(Image.open(mask_file))
show_image(mask_img)

# +

gen_mask_file = "/storage/ice1/0/3/sprathipati6/fml/proj/Painter/SegGPT/SegGPT_inference/mask_example.png"

# Load this supposedly best mask
mask_img = np.array(Image.open(gen_mask_file))
show_image(mask_img)

check_mask = True
if check_mask:
    unique_values, counts = np.unique(mask_img, return_counts=True)
    print(dict(zip(unique_values, counts)))
    
binary_mask = (mask_img > 0).astype(np.uint8) * 255
show_image(binary_mask)

if check_mask:
    unique_values, counts = np.unique(binary_mask, return_counts=True)
    print(dict(zip(unique_values, counts)))


# +
mask_file = "/storage/ice1/0/3/sprathipati6/fml/proj/Painter/SegGPT/SegGPT_inference/output_masks/25/output_mask.png"

mask_img = np.array(Image.open(gen_mask_file))
a = mask_img[:,:,0]
b = mask_img[:,:,1]
c = mask_img[:,:,2]

show_image(a)
show_image(b)
show_image(c) 

""


###############################################################################
# ### 



""


""
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from random import choice
import numpy as np
from IPython.display import clear_output, display
import os

def show_image(img, t):
    plt.imshow(img)
    plt.axis('off')
    plt.title(t)
    plt.show()
    
def render(path, t):
    arr = np.array(Image.open(path))
    show_image(arr, t)

samples = np.load('/storage/ice1/0/3/sprathipati6/fml/proj/samples.npy', allow_pickle=True)
labels = np.load('/storage/ice1/0/3/sprathipati6/fml/proj/labels.npy', allow_pickle=True)

# all_my_0s = [14, 16, 20, 27, 39, 41, 49, 53, 55, 58, 59, 75, 76, 78, 86, 93, 105, 112, 117, 135, 138, 144, 147, 149, 167, 169, 183, 189, 190, 203, 235, 236, 238, 239, 243, 249, 254, 275, 278, 279, 282, 289, 347, 361, 370, 376, 398]
# all_my_bad = [4, 6, 12, 26, 34, 37, 62, 68, 84, 97, 98, 111, 118, 120, 146, 162, 173, 200, 219, 221, 224, 265, 277, 284, 288, 302, 315, 336, 355, 372, 374, 375, 379, 384, 394]
all_my_good = [0, 1, 2, 3, 7, 8, 15, 17, 25, 32, 36, 40, 43, 48, 51, 52, 60, 61, 65, 69, 71, 72, 74, 79, 80, 89, 90, 92, 94, 96, 100, 101, 104, 106, 107, 109, 113, 114, 119, 121, 122, 124, 125, 127, 129, 130, 131, 132, 133, 136, 137, 140, 141, 142, 143, 145, 148, 150, 151, 152, 153, 154, 155, 159, 170, 171, 172, 174, 175, 176, 178, 181, 187, 191, 193, 194, 195, 196, 202, 205, 206, 207, 208, 209, 210, 211, 214, 215, 216, 217, 218, 220, 222, 226, 228, 229, 230, 231, 233, 234, 237, 240, 241, 245, 252, 253, 259, 262, 263, 264, 266, 267, 268, 269, 270, 272, 273, 274, 276, 280, 281, 283, 286, 287, 291, 293, 295, 296, 297, 298, 299, 300, 303, 304, 305, 306, 309, 310, 313, 316, 317, 319, 320, 321, 322, 323, 325, 328, 333, 339, 340, 341, 342, 343, 345, 348, 350, 351, 352, 353, 354, 358, 363, 364, 365, 366, 367, 373, 377, 378, 381, 385, 386, 387, 388, 390, 395, 396, 397, 399]

all_my = [74, 79, 80, 89, 90, 92, 94, 96, 100, 101, 104, 106, 107, 109, 113, 114, 119, 121, 122, 124, 125, 127, 129, 130, 131, 132, 133, 136, 137, 140, 141, 142, 143, 145, 148, 150, 151, 152, 153, 154, 155, 159, 170, 171, 172, 174, 175, 176, 178, 181, 187, 191, 193, 194, 195, 196, 202, 205, 206, 207, 208, 209, 210, 211, 214, 215, 216, 217, 218, 220, 222, 226, 228, 229, 230, 231, 233, 234, 237, 240, 241, 245, 252, 253, 259, 262, 263, 264, 266, 267, 268, 269, 270, 272, 273, 274, 276, 280, 281, 283, 286, 287, 291, 293, 295, 296, 297, 298, 299, 300, 303, 304, 305, 306, 309, 310, 313, 316, 317, 319, 320, 321, 322, 323, 325, 328, 333, 339, 340, 341, 342, 343, 345, 348, 350, 351, 352, 353, 354, 358, 363, 364, 365, 366, 367, 373, 377, 378, 381, 385, 386, 387, 388, 390, 395, 396, 397, 399]
# all_my_all_good = [0, 1, 2, 3, 7, 8, 15, 17, 25, 32, 36, 40, 43, 48, 51, 52, 60, 61, 65, 69, 71, 72, 74, 79, 80, 89, 90, 92, 94, 96, 100, 101, 104, 106, 107, 109, 113, 114, 119, 121, 122, 124, 125, 127, 129, 130, 131, 132, 133, 136, 137, 140, 141, 142, 143, 145, 148, 150, 151, 152, 153, 154, 155, 159, 170, 171, 172, 174, 175, 176, 178, 181, 187, 191, 193, 194, 195, 196, 202, 205, 206, 207, 208, 209, 210, 211, 214, 215, 216, 217, 218, 220, 222, 226, 228, 229, 230, 231, 233, 234, 237, 240, 241, 245, 252, 253, 259, 262, 263, 264, 266, 267, 268, 269, 270, 272, 273, 274, 276, 280, 281, 283, 286, 287, 291, 293, 295, 296, 297, 298, 299, 300, 303, 304, 305, 306, 309, 310, 313, 316, 317, 319, 320, 321, 322, 323, 325, 328, 333, 339, 340, 341, 342, 343, 345, 348, 350, 351, 352, 353, 354, 358, 363, 364, 365, 366, 367, 373, 377, 378, 381, 385, 386, 387, 388, 390, 395, 396, 397, 399]

# all_my_all_good = [0, 1, 2, 3, 7, 8, 15, 17, 25, 32, 36, 40, 43, 48, 51, 52, 60, 61, 65, 69, 71, 72, 74, 79, 80, 89, 90, 92, 94, 96, 100, 101, 104, 106, 107, 109, 113, 114, 119, 121, 122, 124, 125, 127, 129, 130, 131, 132, 133, 136, 137, 140, 141, 142, 143, 145, 148, 150, 151, 152, 153, 154, 155, 159, 170, 171, 172, 174, 175, 176, 178, 181, 187, 191, 193, 194, 195, 196, 202, 205, 206, 207, 208, 209, 210, 211, 214, 215, 216, 217, 218, 220, 222, 226, 228, 229, 230, 231, 233, 234, 237, 240, 241, 245, 252, 253, 259, 262, 263, 264, 266, 267, 268, 269, 270, 272, 273, 274, 276, 280, 281, 283, 286, 287, 291, 293, 295, 296, 297, 298, 299, 300, 303, 304, 305, 306, 309, 310, 313, 316, 317, 319, 320, 321, 322, 323, 325, 328, 333, 339, 340, 341, 342, 343, 345, 348, 350, 351, 352, 353, 354, 358, 363, 364, 365, 366, 367, 373, 377, 378, 381, 385, 386, 387, 388, 390, 395, 396, 397, 399]
all_my_all_good = [0, 1, 2, 3, 7, 8, 9, 11, 13, 15, 17, 18, 19, 21, 22, 24, 25, 28, 29, 31, 32, 35, 36, 40, 43, 44, 45, 46, 48, 51, 52, 57, 60, 61, 63, 64, 65, 67, 69, 71, 72, 73, 74, 77, 79, 80, 81, 85, 89, 90, 92, 94, 96, 99, 100, 101, 102, 103, 104, 106, 107, 109, 113, 114, 116, 119, 121, 122, 123, 124, 125, 127, 128, 129, 130, 131, 132, 133, 136, 137, 139, 140, 141, 142, 143, 145, 148, 150, 151, 152, 153, 154, 155, 156, 158, 159, 160, 161, 165, 166, 170, 171, 172, 174, 175, 176, 177, 178, 179, 180, 181, 182, 185, 186, 187, 188, 191, 193, 194, 195, 196, 197, 198, 199, 202, 205, 206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 220, 222, 226, 228, 229, 230, 231, 233, 234, 237, 240, 241, 242, 244, 245, 246, 247, 248, 251, 252, 253, 255, 256, 257, 258, 259, 260, 262, 263, 264, 266, 267, 268, 269, 270, 271, 272, 273, 274, 276, 280, 281, 283, 285, 286, 287, 290, 291, 292, 293, 295, 296, 297, 298, 299, 300, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 330, 331, 332, 333, 335, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 348, 349, 350, 351, 352, 353, 354, 356, 358, 359, 363, 364, 365, 366, 367, 371, 373, 377, 378, 380, 381, 383, 385, 386, 387, 388, 390, 393, 395, 396, 397, 399]

all_my_all_bad = [4, 6, 10, 12, 23, 26, 30, 34, 37, 42, 47, 54, 56, 62, 68, 82, 83, 84, 87, 88, 97, 98, 108, 111, 115, 118, 120, 134, 146, 162, 164, 168, 173, 200, 201, 204, 219, 221, 224, 225, 227, 261, 265, 277, 284, 288, 301, 302, 315, 329, 336, 355, 368, 372, 374, 375, 379, 384, 389, 391, 394]


print(len(all_my_all_good))
print(len(all_my_all_bad))


mask_dir = "/home/hice1/sprathipati6/scratch/fml/proj/Prompting results/Bus/st1/masks"
all_mask_files = os.listdir(mask_dir)

for IDX in all_my:
    output_imgs_f = f"/storage/ice1/0/3/sprathipati6/fml/proj/Painter/SegGPT/SegGPT_inference/output_images/{IDX}/output_img.png"
    output_masks_f = f"/storage/ice1/0/3/sprathipati6/fml/proj/Painter/SegGPT/SegGPT_inference/output_masks/{IDX}/output_mask.png"

    show_image(samples[IDX], f"Original img #{IDX}")

    show_image(labels[IDX], f"Label #{IDX}")

    # render(output_imgs_f, f"Output img #{IDX}")

    render(output_masks_f, f"Mask {IDX}")
    
    filtered_sorted_files = sorted(
    [f for f in all_mask_files if f.startswith(f'{IDX}_') and f.endswith('_mask.png')],
    key=lambda x: int(x.split('_')[1])) 

    if (len(filtered_sorted_files) != 0):
        last_file = filtered_sorted_files[-1] 
        ph0_mask = np.array(Image.open(os.path.join(mask_dir, last_file)))
        
    show_image(ph0_mask, f"Phase 0 Output")
    
    input()
    clear_output(wait=True)
    
    



""


import cv2
from PIL import Image
import matplotlib.pyplot as plt
from random import choice
import numpy as np
from IPython.display import clear_output, display
import os

def show_image(img, t):
    plt.imshow(img)
    plt.axis('off')
    plt.title(t)
    plt.show()
    
def render(path, t):
    arr = np.array(Image.open(path))
    show_image(arr, t)

IDX = 4
    
samples = np.load('/storage/ice1/0/3/sprathipati6/fml/proj/samples.npy', allow_pickle=True)
labels = np.load('/storage/ice1/0/3/sprathipati6/fml/proj/labels.npy', allow_pickle=True)

mask_dir = "/home/hice1/sprathipati6/scratch/fml/proj/Prompting results/Bus/st1/masks"
all_mask_files = os.listdir(mask_dir)

output_imgs_f = f"/storage/ice1/0/3/sprathipati6/fml/proj/Painter/SegGPT/SegGPT_inference/output_images/{IDX}/output_img.png"
output_masks_f = f"/storage/ice1/0/3/sprathipati6/fml/proj/Painter/SegGPT/SegGPT_inference/output_masks/{IDX}/output_mask.png"

filtered_sorted_files = sorted(
[f for f in all_mask_files if f.startswith(f'{IDX}_') and f.endswith('_mask.png')],
key=lambda x: int(x.split('_')[1])) 

if (len(filtered_sorted_files) != 0):
    last_file = filtered_sorted_files[-1] 
    ph0_mask = np.array(Image.open(os.path.join(mask_dir, last_file)))

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].imshow(samples[IDX])
axes[0, 0].axis('off')  # Turn off the axis
axes[0, 0].set_title('Original Image')

axes[0, 1].imshow(labels[IDX])
axes[0, 1].axis('off')
axes[0, 1].set_title('Ground Truth')

axes[1, 0].imshow(np.array(Image.open(output_masks_f)))
axes[1, 0].axis('off')
axes[1, 0].set_title('SegGPT Mask')

axes[1, 1].imshow(ph0_mask)
axes[1, 1].axis('off')
axes[1, 1].set_title('Manual prompting Mask')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplots to fit into figure area.
plt.show()

# Ask the user for input to name the file
filename = input("Please enter a filename to save the image: ")

if filename != "lite":
    # Add the appropriate file extension if not provided
    if not filename.lower().endswith('.png'):
        filename += '.png'

    # Save the figure
    fig.savefig(filename)
    print(f"Image saved as {filename}")

