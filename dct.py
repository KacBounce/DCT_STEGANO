import cv2
import numpy as np
import random
import threading



# Load the stego-image
image = cv2.imread('Lenna(test_image).png', cv2.IMREAD_GRAYSCALE)  # Load as grayscale 
hidden = cv2.imread('hidden.jpg', cv2.IMREAD_GRAYSCALE)

dct_image = cv2.dct(np.float32(image))

# Define the quantization matrix for the Y component
# This matrix depends on the JPEG quality setting
# For example, for a quality setting of 50 (typical value), you can use a standard JPEG quantization matrix
quality = 50
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])

DQ = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 41, 60, 55],
        [14, 13, 16, 17, 28, 40, 48, 56],
        [14, 17, 22, 20, 36, 61, 56, 43],
        [18, 22, 26, 39, 48, 76, 72, 54],
        [24, 25, 39, 45, 57, 73, 79, 64],
        [49, 64, 55, 61, 72, 85, 84, 71],
        [72, 92, 95, 69, 78, 70, 72, 70]])

# Split the image into 8x8 blocks (assuming the image dimensions are multiples of 8)
block_size = 8
num_blocks_height = image.shape[0] // block_size
num_blocks_width = image.shape[1] // block_size

def zero_length(set):
    length = 0
    for i in set:
        if (i == 0 or i == -0):
            length += 1
        else:
            return length
    return length

def highest_non_zero(set):
    for i in range(len(set)):
        if (set[i] != 0):
            index = i
    return index

def all_zeros(set):
    for i in range(len(set)):
        if (set[i] != 0):
            return False
    return True


def get_blocks(image, do):
    blocks = []

    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            # Extract an 8x8 block
            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            
            # Perform DCT on the block
            dct_block = cv2.dct(np.float32(block))
            if (do):
                quantized_dct_block = np.round(dct_block / (DQ * (quality / 50)))
            else:            
                # Quantize the DCT coefficients using the quantization matrix
                quantized_dct_block = np.round(dct_block / (Q * (quality / 50)))

            blocks.append(quantized_dct_block)
    return blocks

def get_sets_from_blocks(blocks): 
    sets = []
    for i in blocks:   
        set1 = []
        set2 = []
        set3 = []
        set4 = []
        set5 = []
        set6 = []
        set7 = []
        set8 = []
        set9 = []
        allSets = {}
        counter = 1
        key = "Set"

        for x in range(0,7):
            set1.append(i[x,x])
        set1.reverse()
        allSets[key+str(counter)] = set1
        counter += 1

        for x in range(0,7):
            set2.append(i[x,x+1])
        set2.reverse()
        allSets[key+str(counter)] = set2
        counter += 1

        for x in range(0,7):
            set3.append(i[x+1,x])
        set3.reverse()
        allSets[key+str(counter)] = set3
        counter += 1

        for x in range(0,6):
            set4.append(i[x,x+2])
        set4.reverse()
        allSets[key+str(counter)] = set4
        counter += 1

        for x in range(0,6):
            set5.append(i[x+2,x])
        set5.reverse()
        allSets[key+str(counter)] = set5
        counter += 1

        for x in range(0,5):
            set6.append(i[x,x+3])
        set6.reverse()
        allSets[key+str(counter)] = set6
        counter += 1

        for x in range(0,5):
            set7.append(i[x+3,x])
        set7.reverse()
        allSets[key+str(counter)] = set7
        counter += 1

        for x in range(0,4):
            set8.append(i[x,x+4])
        set8.reverse()
        allSets[key+str(counter)] = set8
        counter += 1

        for x in range(0,4):
            set9.append(i[x+4,x])
        set9.reverse()
        allSets[key+str(counter)] = set9
        counter += 1
        
        sets.append(allSets)

    return sets

def dequantize_blocks(blocks):
    dequantized_blocks = []

    # Loop through the quantized blocks and dequantize them
    for quantized_block in blocks:
        # Dequantize the block using the quantization matrix and quality factor
        dequantized_block = quantized_block * (DQ * (quality / 50))
        
        # Perform the inverse DCT on the dequantized block
        dequantized_block = cv2.idct(np.float32(dequantized_block))
        
        # Convert to integers and clip to valid pixel values
        dequantized_block = np.uint8(np.clip(dequantized_block, 0, 255))
        
        # Append the dequantized block to the list
        dequantized_blocks.append(dequantized_block)

    return dequantized_blocks

def get_image_from_blocks(blocks):
    # Assuming 'blocks' is a list of 8x8 blocks (e.g., as obtained from DCT or other operations)
    # Define the dimensions of the image and initialize it
    image_height, image_width = num_blocks_height * block_size, num_blocks_width * block_size
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.uint8)

    # Reconstruct the image from blocks
    block_index = 0
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            # Get the current block
            block = blocks[block_index]
            
            # Place the block in the reconstructed image
            reconstructed_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = block
            
            block_index += 1

    # 'reconstructed_image' now contains the image reconstructed from the blocks
    return reconstructed_image

def hide_message(sets):
    set_array = []
    binary_message = "010101010110101101110010011110010111010001100001001000000111011101101001011000010110010001101111011011010110111101110011011000110010000001101110011100100010000000110001"
    binary_array = [int(bit) for bit in binary_message]
    counter = 0

    for i in sets:
        for j in i:
            set_array.append(i[j])
    
    
    for i in set_array:
        if (not all_zeros(i)):
            if (counter < len(binary_array)):
                last_zero = zero_length(i)
                if (last_zero >= 2):
                    if (binary_array[counter] == 1):
                        secret = random.randint(0, 1)
                        i[last_zero - 2] = 1 if secret == 1 else -1
                    else:
                        i[last_zero - 2] = 0
                    counter = counter + 1
                    
    return set_array

def encrypt():  
    blocks = get_blocks(image, 0)  
    sets = get_sets_from_blocks(blocks)
    
    counter = 0

    set_array = hide_message(sets)
    
    for i in range(10):
        print(sets[i])

    for i in blocks:
        set_array[counter].reverse()
        for x in range(0,7):
            i[x,x] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,7):
            i[x,x+1] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,7):
            i[x+1,x] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,6):
            i[x,x+2] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,6):
            i[x+2,x] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,5):
            i[x,x+3] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,5):
            i[x+3,x] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,4):
            i[x,x+4] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,4):
            i[x+4,x] = set_array[counter][x]
        counter = counter + 1  
        
    for i in range(5):
        print(blocks[i])

    dequantized_blocks = dequantize_blocks(blocks)
    image_with_info = get_image_from_blocks(dequantized_blocks)
    dct_image_with_info = cv2.dct(np.float32(image_with_info))

    cv2.imshow('hidden', image_with_info)

    cv2.imshow('dct normal', dct_image)
    cv2.imshow('dct hidden', dct_image_with_info)
    cv2.imshow('diff', image - image_with_info)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

encrypt()

def get_message(sets):
    received_message = ""
    set_array = []

    for i in sets:
        #print(i)z
        for j in i:
            set_array.append(i[j])
            
    for i in range(5):
        print(sets[i])
    x = 0
    for i in set_array:    
        if (x < 155):
            if (not all_zeros(i)):
                index = highest_non_zero(i)
                print(index,"",len(i),i[index])
                if (index < len(i) - 1):
                    print(i[index]," ",i[index+1])
                    if ((i[index] == 1 or i[index] == -1) and i[index + 1] == 0):
                        received_message += "1"
                        x += 1
                elif (((i[index - 1] == 1 or i[index - 1] == -1) and i[index] != 0 and i[index - 3] == 0)
                    or ((i[index - 1] != 1 and i[index - 1] != -1) and i[index - 2] and i[index - 3] == 0)):
                    received_message += "0"
                    x += 1
            else:
                received_message += "0"
                x += 1

    return received_message

            
        


def decrypt():
    blocks = get_blocks(hidden, 1)
    for i in range(5):
        print(blocks[i])
    sets = get_sets_from_blocks(blocks)
    

    x = get_message(sets)

    print("WHAT: ", x)

decrypt()
