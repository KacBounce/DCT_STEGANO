import cv2
import numpy as np
from dct import *
import random
import threading

max_message_bits = 8
local_max = 35
local_curr_sets = []
local_set_indexes = []

img1 = cv2.imread('C:\\Users\\kbednarski\\OneDrive - Anegis\\Desktop\\Inzynierka\\Lenna(test_image).png', cv2.IMREAD_GRAYSCALE)
lock = threading.Lock()

def get_image_from_blocks(blocks):
    num_blocks_height = img1.shape[0] // block_size
    num_blocks_width = img1.shape[1] // block_size
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

def hide(set_array, binary_array, msg_length_bin):
    x = 0
    counter = 0
    len_counter = 0
    for i in set_array:
        if (counter < len(binary_array)):
            last_zero = zero_length(i)
            
            #condition A
            if((i[last_zero - 1] == 1 or i[last_zero - 1] == -1) and i[last_zero] == 0):
                        if (i[last_zero - 1] > 0):
                            i[last_zero - 1] += 1
                        else:
                            i[last_zero - 1] -= 1   

            #condition B
            if ((i[0] == -1 or i[0] == 1) and i[1] == 0):
                if (i[0] > 0):
                    i[0] += 1
                else:
                    i[0] -= 1

            #condition C
            if ((i[1] == -1 or i[1] == 1) and i[0] == 0 and i[2] == 0):
                if (i[1] > 0):
                    i[1] += 1
                else:
                    i[1] -= 1
                    
            #hiding the message length
            if (x < max_message_bits - 2):
                if (x < max_message_bits - len(msg_length_bin) - 1):
                    i[last_zero - 2] = 0
                    x += 1
                else:
                    if (last_zero >= 2 and len_counter < len(msg_length_bin) - 1):                              
                        if (msg_length_bin[len_counter] == 1):
                            secret = random.randint(0, 1)
                            i[last_zero - 2] = 1 if secret == 1 else -1
                        else:
                            i[last_zero - 2] = 0
                        len_counter = len_counter + 1
                        x += 1
            else:
                #Main data hiding   
                if (last_zero >= 2):                      
                    if (binary_array[counter] == 1):
                        secret = random.randint(0, 1)
                        i[last_zero - 2] = 1 if secret == 1 else -1
                    else:
                        i[last_zero - 2] = 0 
                    counter = counter + 1
    return set_array

def pso():
    global local_max, local_curr_sets, local_set_indexes, lock
    message = "No hej kurczaki"
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_array = [int(bit) for bit in binary_message]
    msg_length = len(binary_array)
    msg_length_bin = [int(bit) for bit in bin(msg_length)[2:]]   

    max_iterations = 100
    for i in range(max_iterations): 
        blocks = get_blocks(img1, 0)  
        sets = get_sets_from_blocks(blocks)
        set_array = []
        for i in sets:
            for j in i:
                set_array.append(i[j]) 

        curr_sets = []
        set_indexes = []
        for i in range(msg_length + max_message_bits):
            index = random.randint(0, len(set_array))   
            curr_sets.append(set_array[index])
            set_indexes.append(index)
        curr_sets = hide(curr_sets, binary_array, msg_length_bin)
        counter = 0
        for i in set_indexes:
            set_array[i] = curr_sets[counter]
            counter += 1
        counter = 0
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

        dequantized_blocks = dequantize_blocks(blocks)
        image_with_info = get_image_from_blocks(dequantized_blocks)
        #dct_image_with_info = cv2.dct(np.float32(image_with_info))

        #cv2.imshow('hidden', image_with_info)
        #cv2.imwrite('hidden.jpg', image_with_info)

        #dct_image = cv2.dct(np.float32(img1))
        #cv2.imshow('dct normal', dct_image)  
        #cv2.imshow('dct hidden', dct_image_with_info)

       # cv2.waitKey(0)

       # cv2.destroyAllWindows()

        psnr = cv2.PSNR(img1, image_with_info)

        if (psnr > local_max):
            lock.acquire()
            local_max = psnr
            local_curr_sets = curr_sets
            local_set_indexes = set_indexes
            lock.release()



t1 = threading.Thread(target=pso)
t2 = threading.Thread(target=pso)
t3 = threading.Thread(target=pso)
t4 = threading.Thread(target=pso)

t1.start()
t2.start()
t3.start()
t4.start()

t1.join()
t2.join()
t3.join()
t4.join()