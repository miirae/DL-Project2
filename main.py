import cv2
import os
import glob
import torch
import numpy as np

torch.set_default_dtype(torch.float32)
# Load dataset
img_dir = "datasets/**/*.jpg"
files = glob.glob(img_dir)
data = []
for f1 in files:
    img = cv2.imread(f1)
    if img is not None:
        resized_img = cv2.resize(img, (128, 128))
        data.append(resized_img)

data_np = np.stack(data)

# convert to a PyTorch tensor
# Since cv2.imread() loads images as BGR, ensure this is what you want, or convert to RGB if needed
data_tensor = torch.tensor(data_np).permute(0, 3, 1, 2)

indices = torch.randperm(data_tensor.size(0))
shuffled_data = data_tensor[indices]


# Augment dattaset
n_images, channels, height, width = data_tensor.shape

augmented_data = torch.empty(n_images * 10, channels, height, width, dtype=torch.float)

for i in range(n_images):
    # tensor to numpy array in HWC format
    original_image = data_tensor[i].permute(1, 2, 0).numpy().astype(np.uint8)

    for j in range(10): # 10 versions of each img
        if np.random.rand() > 0.5: #50% chance flip
            augmented_image = cv2.flip(original_image, 1)
        else:
            augmented_image = original_image.copy()

        #  random cropping
        x_start = np.random.randint(0, width // 10)
        y_start = np.random.randint(0, height // 10)
        x_end = np.random.randint(9 * width // 10, width)
        y_end = np.random.randint(9 * height // 10, height)
        cropped_image = augmented_image[y_start:y_end, x_start:x_end]
        resized_image = cv2.resize(cropped_image, (width, height))

        scale_factor = np.random.uniform(0.6, 1.0)
        augmented_image = np.clip(resized_image * scale_factor, 0, 255).astype(np.uint8)

        # Convert back to PyTorch tensor and add to augmented_data
        augmented_data[i * 10 + j] = torch.tensor(resized_image).permute(2, 0, 1).float()

# convert images to L * a * b* color space
n_images, channels, height, width = augmented_data.shape


augmented_dir = 'augmented'
os.makedirs(augmented_dir, exist_ok=True)
os.makedirs('L', exist_ok=True)
os.makedirs('a', exist_ok=True)
os.makedirs('b', exist_ok=True)

# save augmented before Lab conversion
for i in range(augmented_data.shape[0]):
    # Convert tensor to numpy array
    img_np = augmented_data[i].permute(1, 2, 0).numpy()
    img_np = img_np.astype(np.uint8)

    # Save the image
    cv2.imwrite(os.path.join(augmented_dir, f"augmented_{i}.jpg"), img_np)




# save augmented after lab conversion
for i in range(augmented_data.shape[0]):
    # Convert tensor to numpy array in BGR format
    img_np = augmented_data[i].permute(1, 2, 0).numpy().astype(np.uint8)

    imageLAB = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)

    # Split the L*a*b* image into L, a, and b channels
    L, a, b = cv2.split(imageLAB)

    # to visualize the a and b channels with color, we'll combine them with a neutral L value (50% gray)
    L_neutral = 128 * np.ones_like(L)  # 50% gray for L

    # Create colored visualizations for the a and b channels
    lab_a_colored = cv2.merge([L_neutral, a, 128 * np.ones_like(b)])  # Neutral values for L and b
    lab_b_colored = cv2.merge([L_neutral, 128 * np.ones_like(a), b])  # Neutral values for L and a

    # Convert back to BGR color space for display
    img_a_colored = cv2.cvtColor(lab_a_colored, cv2.COLOR_LAB2BGR)
    img_b_colored = cv2.cvtColor(lab_b_colored, cv2.COLOR_LAB2BGR)

    cv2.imwrite(f'a/aug_a_{i}.jpg', img_a_colored)
    cv2.imwrite(f'b/aug_b_{i}.jpg', img_b_colored)


'''
for i in range(n_images):
    image_rgb = augmented_data[i].numpy().transpose(1, 2, 0).astype(np.uint8)
    imageLAB = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    L, a, b = cv2.split(imageLAB)

    cv2.imwrite(f'augmented/img_{i}.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'L/img_{i}_L.jpg', L)
'''










#cv2.imshow(’image’, imageLAB)
#L,a,b=cv2.split(imageLAB)
#cv2.imshow(’L’,L)
#cv2.imshow(’a’,a)
#cv2.imshow(’b’,b)

'''

# Create directories for saving images
os.makedirs('augmented/L', exist_ok=True)
os.makedirs('augmented/a', exist_ok=True)
os.makedirs('augmented/b', exist_ok=True)

for i in range(n_images):
    # Convert the tensor to a numpy array and back to RGB for OpenCV
    image_rgb = augmented_data[i].permute(1, 2, 0).numpy().astype(np.uint8)

    # Convert from RGB to L*a*b* color space
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)

    # Split the L*a*b* image into L, a, and b components
    L, a, b = cv2.split(image_lab)

    # Save the L, a, and b components
    cv2.imwrite(f'augmented/L/img_{i}_L.jpg', L)
    cv2.imwrite(f'augmented/a/img_{i}_a.jpg', a)
    cv2.imwrite(f'augmented/b/img_{i}_b.jpg', b)

L = cv2.imread('augmented/L/img_0_L.jpg', cv2.IMREAD_GRAYSCALE)
a = cv2.imread('augmented/a/img_0_a.jpg', cv2.IMREAD_GRAYSCALE)
b = cv2.imread('augmented/b/img_0_b.jpg', cv2.IMREAD_GRAYSCALE)

# Display the L, a, and b images
cv2.imshow('L Channel', L)
cv2.imshow('a Channel', a)
cv2.imshow('b Channel', b)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


'''
# testing channels with one image

img = cv2.imread('datasets/face_images/image00000.jpg')  # Replace 'path_to_your_image.jpg' with your actual image path


# Convert the image to L*a*b* color space
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Split the L*a*b* image into L, a, and b channels
L, a, b = cv2.split(img_lab)

# Display the L channel
cv2.imshow('L Channel', L)

# To visualize the a and b channels with color, we'll combine them with a neutral L value (50% gray)
L_neutral = 128 * np.ones_like(L)  # 50% gray for L

# Create colored visualizations for the a and b channels
lab_a_colored = cv2.merge([L_neutral, a, 128 * np.ones_like(b)])  # Neutral values for L and b
lab_b_colored = cv2.merge([L_neutral, 128 * np.ones_like(a), b])  # Neutral values for L and a

# Convert back to BGR color space for display
img_a_colored = cv2.cvtColor(lab_a_colored, cv2.COLOR_LAB2BGR)
img_b_colored = cv2.cvtColor(lab_b_colored, cv2.COLOR_LAB2BGR)

# Display the colored a and b channels
cv2.imshow('a Channel Colored', img_a_colored)
cv2.imshow('b Channel Colored', img_b_colored)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''