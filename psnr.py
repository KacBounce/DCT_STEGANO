import cv2
import numpy as np
from dct import *
import random

max_message_bits = 8

def hide(set_array, binary_array, msg_length_bin):
    x = 0
    counter = 0
    len_counter = 0
    print(msg_length_bin)
    print(binary_array)
    for i in set_array:
        print(i)
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
        print(i)
    return set_array

def pso():
    message = "No hej kurczaki"
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_array = [int(bit) for bit in binary_message]
    msg_length = len(binary_array)
    msg_length_bin = [int(bit) for bit in bin(msg_length)[2:]]   

    img1 = cv2.imread('C:\\Users\\kbednarski\\OneDrive - Anegis\\Desktop\\Inzynierka\\Lenna(test_image).png', cv2.IMREAD_GRAYSCALE)

    max_iterations = 1
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
        print(len(set_array))
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
        dct_image_with_info = cv2.dct(np.float32(image_with_info))

        save_path = filedialog.asksaveasfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])

        cv2.imshow('hidden', image_with_info)
        cv2.imwrite(save_path, image_with_info)

        dct_image = cv2.dct(np.float32(img1))
        cv2.imshow('dct normal', dct_image) 
        cv2.imshow('dct hidden', dct_image_with_info)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

        psnr = cv2.PSNR(img1, image_with_info)

        print(psnr,"dB")

pso()
