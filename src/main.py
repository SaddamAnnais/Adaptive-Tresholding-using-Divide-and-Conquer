import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def mean_threshold(image, k=0.85):
    image_np = np.array(image)
    return np.mean(image_np)*k


def adaptive_treshold(block):
    # Compute treshold
    threshold = mean_threshold(block)

    # Adaptive tresholding for a blok of image
    block = block > threshold

    # Change the dtype uint8
    block = block.astype(np.uint8) * 255

    return block


def divide_and_conquer(blocks, func):
    # divide and conquer approach to apply a function to each block

    n_blocks = len(blocks)
    if n_blocks == 1:
        filtered_blocks = [func(blocks[0])]
    else:
        mid = n_blocks // 2
        left_blocks = divide_and_conquer(blocks[:mid], func)
        right_blocks = divide_and_conquer(blocks[mid:], func)
        filtered_blocks = left_blocks + right_blocks

    return filtered_blocks


def enhance_image(image_path, block_size):
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
    filtered_blocks = divide_and_conquer(blocks, adaptive_treshold)

    # merge the blocks back together
    enhanced_image = Image.new('L', (width, height))
    index = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            enhanced_block = filtered_blocks[index]
            enhanced_image.paste(Image.fromarray(enhanced_block), (j, i))
            index += 1

    # return the enhanced image
    return enhanced_image


if __name__ == '__main__':
    # example usage
    adaptive_image = enhance_image('test/test2.jpg', 16)
    adaptive_image.show()

    image = Image.open('test/test2.jpg').convert('L')
    global_treshold = adaptive_treshold(image)
    Image.fromarray(global_treshold).show()
