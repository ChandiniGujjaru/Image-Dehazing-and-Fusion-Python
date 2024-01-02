import cv2
import numpy as np

def dark_channel(image, patch_size=15):
    b, g, r = cv2.split(image)
    min_channel = cv2.min(b, cv2.min(g, r))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def estimate_atmosphere(image, dark_channel, top_percent=0.01):
    flat_dark = dark_channel.flatten()
    num_pixels = flat_dark.shape[0]
    num_brightest = int(top_percent * num_pixels)
    indices = flat_dark.argsort()[-num_brightest:]
    bright_pixels = image.reshape(-1, 3)[indices]
    atmosphere = np.mean(bright_pixels, axis=0)
    return atmosphere

def dehaze(image, t, atmosphere, t0=0.2):
    t = np.maximum(t, t0)
    dehazed_image = np.empty_like(image, dtype=np.uint8)
    for i in range(3):
        dehazed_image[:, :, i] = ((image[:, :, i] - atmosphere[i]) / t) + atmosphere[i]
    return dehazed_image

def apply_gamma_correction(image, gamma=1.8):
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    gamma_corrected = np.uint8(gamma_corrected)
    return gamma_corrected

def main():
    image = cv2.imread("C:\\datasets\\dataset5.jpg")
    
    # Parameter values
    patch_size = 15
    top_percent = 0.01
    t0 = 0.2
    alpha = 0.8  # Weight for image fusion
    
    dark = dark_channel(image, patch_size)
    atmosphere = estimate_atmosphere(image, dark, top_percent)
    
    transmission = 1.0 - dark / 255.0
    
    dehazed_image = dehaze(image, transmission, atmosphere, t0)
    
    # Apply gamma correction to the dehazed image
    gamma_value = 2.0  # You can adjust this gamma value as needed
    dehazed_image = apply_gamma_correction(dehazed_image, gamma=gamma_value)
    
    # Image fusion
    fused_image = cv2.addWeighted(dehazed_image, alpha, image, 1 - alpha, 0)
    
    cv2.imwrite("dehazed_image_gamma5.jpeg", dehazed_image)
    cv2.imwrite("fused_image_gamma5.jpeg", fused_image)
    cv2.imwrite("dark_channel5.jpeg", dark)

if __name__ == "__main__":
    main()
