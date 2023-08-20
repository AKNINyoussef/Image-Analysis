import numpy as np
import cv2
import matplotlib.pyplot as plt

####################################################################
                        # Image Enhancement #
####################################################################

def Picture_Enhancement(Any_Image):

# Calling the image, and converting it to grayscale
    Pic_01 = cv2.imread(Any_Image)
    Pic_Gray = cv2.cvtColor(Pic_01, cv2.COLOR_BGR2GRAY)

# Displaying the Original Image
    plt.figure()
    plt.imshow(Pic_01)
    plt.title('Original Pic')

# Min and Max intensity values of the original image
    hist_min = np.min(Pic_Gray)
    hist_max = np.max(Pic_Gray)
    print('max intensity value of original picture = ', hist_max)
    print('min intensity value of original picture = ', hist_min)

# visualizing histogram of the grayscaled image
    plt.figure()
    plt.hist(Pic_Gray.ravel(), bins=256, range=(0, 256))
    plt.xticks(range(0, 256, 20))
    plt.title('Histogram_Pic')
    plt.xlabel('Intensity value (I)')
    plt.ylabel('Number of Pixels (N)')
    plt.show()

# Enhance the image using contrast stretching and visualize it
    new_gray_value = ((Pic_Gray - hist_min) / (hist_max - hist_min)) * 255       #new values of contrast streching
    new_gray_value = np.uint8(new_gray_value)

    plt.figure()
    plt.imshow(np.array(new_gray_value), cmap='gray')
    plt.title('Enhanced Image')

    plt.figure()
    plt.hist(np.array(new_gray_value).ravel(), bins=256, range=(0, 256))
    plt.xticks(range(0, 256, 20))
    plt.title('Histogram of Enhanced Image')
    plt.xlabel('Intensity value (I)')
    plt.ylabel('Number of Pixels (N)')
    plt.show()
    return np.array(new_gray_value)

####################################################################
                            # Binarization #
####################################################################

def Binarized_picture(modified_img):

# Comparing the enhanced picture intensity values to a theshold value
# Using different threshold values for and visualize each

    Binary_pic = modified_img.copy()
    Threshold_values = [110, 90, 60, 85]
    for T in Threshold_values:
        for i in range(modified_img.shape[0]):
            for j in range(modified_img.shape[1]):
                if modified_img[i, j] > T:
                    Binary_pic[i, j] = 0
                else:
                    Binary_pic[i, j] = 255

# Visualizing the Binarized picture
        plt.figure()
        plt.imshow(Binary_pic, cmap='gray')
        plt.title(f'Binary Pic T = {T}')
        plt.show()

    return Binary_pic

####################################################################
                #Morphological Operators#
####################################################################

def Morpho_Picture(binary_img):

# We create a structuring element for the morphological operation (15x15 rectangular shape)
    Structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))

# We did the openning operation on our binary image
    img_openning = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, Structuring_element)

# Now we perform closing operation on our opened image
    img_closing = cv2.morphologyEx(img_openning, cv2.MORPH_CLOSE, Structuring_element)

# Visualizing both Binary masked (closed, opened) images
    plt.figure()
    plt.imshow(img_openning, cmap='gray')
    plt.title('opened img')
    plt.figure()
    plt.imshow(img_closing, cmap='gray')
    plt.title('closed img')
    plt.show()
    return img_closing

def Overlayed_img(modified_img, img_closing):

#for overlaying we used (add) function
    Overlayed_img = cv2.add(modified_img,img_closing)
    plt.figure()
    plt.imshow(Overlayed_img, cmap='gray')
    plt.title('Overlayed Image')
    plt.show()
    return Overlayed_img

def main():
# input the path of the image
    input_img_path = r'input_sat_image.JPG' 

# Picture enhancement
    modified_img = Picture_Enhancement(input_img_path)

# Binarize and perform the Morphological operation
    binary_img = Binarized_picture(modified_img)
    img_closing = Morpho_Picture(binary_img)

# Overlay the enhanced image morphologically filtered masked image
    Overlayed_img(modified_img, img_closing)

main()




