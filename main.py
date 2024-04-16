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

    L, a, b = cv2.split(imageLAB)

    # to visualize the a and b channels with color, we'll combine them with a neutral L value (50% gray)
    L_neutral = 128 * np.ones_like(L)  # 50% gray for L

    lab_a_colored = cv2.merge([L_neutral, a, 128 * np.ones_like(b)])
    lab_b_colored = cv2.merge([L_neutral, 128 * np.ones_like(a), b])
    # convert back to BGR color space for display
    img_a_colored = cv2.cvtColor(lab_a_colored, cv2.COLOR_LAB2BGR)
    img_b_colored = cv2.cvtColor(lab_b_colored, cv2.COLOR_LAB2BGR)

    cv2.imwrite(f'a/aug_a_{i}.jpg', img_a_colored)
    cv2.imwrite(f'b/aug_b_{i}.jpg', img_b_colored)










