import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter, morphology


def mean_threshold(image, k=0.85):
    image_np = np.array(image)
    return np.mean(image_np)*k


def adaptive_treshold(block, k):
    # Compute treshold
    threshold = mean_threshold(block, k)

    # Adaptive tresholding for a blok of image
    block = block > threshold

    # Change the dtype uint8
    block = block.astype(np.uint8) * 255

    return block


def divide_and_conquer(blocks, func, k):
    # divide and conquer approach to apply a function to each block

    n_blocks = len(blocks)
    if n_blocks == 1:
        filtered_blocks = [func(blocks[0], k)]
    else:
        mid = n_blocks // 2
        left_blocks = divide_and_conquer(blocks[:mid], func, k)
        right_blocks = divide_and_conquer(blocks[mid:], func, k)
        filtered_blocks = left_blocks + right_blocks

    return filtered_blocks


def enhance_image(image_path, block_size, k):
    # load the image
    image = Image.open(image_path).convert('L')

    width, height = image.size

    # divide the image into blocks
    blocks = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image.crop((j, i, j+block_size, i+block_size))
            blocks.append(block)

    # apply a filter to each block in the frequency domain using divide and conquer
    filtered_blocks = divide_and_conquer(blocks, adaptive_treshold, k)

    # merge the blocks back together
    enhanced_image = Image.new('L', (width, height))
    index = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            enhanced_block = filtered_blocks[index]
            enhanced_image.paste(Image.fromarray(enhanced_block), (j, i))
            index += 1

    # Perform morphological closing operation on the enhanced image
    # enhanced_image = Image.fromarray(morphology.binary_closing(enhanced_image, structure=np.ones((2, 2))))

    # return the enhanced image
    return enhanced_image


if __name__ == '__main__':
    # example usage
    # Proposed Method
    adaptive_image = enhance_image('test/experiments/test3.jpg', 32, 0.95)
    adaptive_image.save('test/result/test3-1.jpg')

    # Otsu's Method
    image1 = cv2.imread('test/experiments/test3.jpg')
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)     
    Image.fromarray(thresh1).save('test/result/test3-2.jpg')
