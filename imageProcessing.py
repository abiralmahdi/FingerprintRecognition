# This script is a simple implementation of feature extraction using Laplacian edge detection.
# The input is an image, and the output is a binary image with the edges detected by the Laplacian operator.

# Import the required libraries for image processing
import cv2

# Define the function that resizes the image to a specific size
def resizeImage(img):
    
    resized = cv2.resize(img, (96, 103), interpolation = cv2.INTER_AREA)
    return resized


# Define the function for feature extraction using Laplacian edge detection
def featureExtractLaplace(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blurring to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Normalize the Laplacian image to 0-255
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Convert the normalized image to a binary image
    binary_image = cv2.adaptiveThreshold(laplacian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return binary_image


# Define the main function that performs the following tasks:
# 1. Read the image from the specified file path
# 2. Resize the image to a specific size (96x103)
# 3. Extract the features using the Laplacian edge detection method
def main():
    # Read the image from the specified file path
    image = cv2.imread("C:/Users/HP/Desktop/Course Materials/6. 5th Semester (Spring-24)/CSE468/Project/SOCOFing/TEST/myLeftIdxFinger.jpg")
    # Resize the image to a specific size (96x103)
    img = resizeImage(image)
    # Extract the features using the Laplacian edge detection method
    features = featureExtractLaplace(img)
    # Note: The output is a binary image with the edges detected by the Laplacian operator.

    # Save the output image to a specified file path
    cv2.imwrite("C:/Users/HP/Desktop/Course Materials/6. 5th Semester (Spring-24)/CSE468/Project/SOCOFing/TEST/100__M_testing.BMP", features)
    
    # Display the output image on the screen
    cv2.imshow("img", features)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()