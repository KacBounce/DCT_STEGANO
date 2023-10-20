import numpy as np
from PIL import Image
import random
import cv2

delimiter = '##END'
delimiter = ''.join(format(ord(char), '08b') for char in delimiter)

def create_chromosome():        
    chromosome = random.randint(0,67108864)
    binary_chromosome = format(chromosome,'b')
    while(len(binary_chromosome)<26):
        binary_chromosome = "0" + binary_chromosome
    return [chromosome, binary_chromosome]

chromosome, binary_chromosome = create_chromosome()
print(chromosome, binary_chromosome)

def transform_secret_image(img, binary_chromosome):
    width, height = img.size
    binary_data = ''.join(format(pixel, '08b') for pixel in img.tobytes())
    if(binary_chromosome[2] == '1'):
        binary_data = ''.join('1' if bit == '0' else '0' for bit in binary_data)
    if(binary_chromosome[1] == '1'):
        binary_data = binary_data[::-1]
    return [binary_data, width, height]

def transform_bits_image(binary_data, width, height, binary_chromosome):
    mode = 'L'  # Grayscale mode 

    # Create an empty image
    img = Image.new(mode, (width, height))
    
    if(binary_chromosome[2] == '1'):
        binary_data = ''.join('1' if bit == '0' else '0' for bit in binary_data)
    if(binary_chromosome[1] == '1'):
        binary_data = binary_data[::-1]

    # Parse the list of bits and set pixel values
    pixels = [int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8)]
    img.putdata(pixels)
    return img

# def Encode(src, binary_message, dest, binary_chromosome):

#     img = Image.open(src, 'r')
#     width, height = img.size
#     array = np.array(list(img.getdata()))

#     if img.mode == 'RGB':
#         n = 3
#     elif img.mode == 'RGBA':
#         n = 4
#     print(len(array))
#     total_pixels = len(array)
#     req_pixels = len(binary_message)
    
#     delimiter = '##END'
#     delimiter = ''.join(format(ord(char), '08b') for char in delimiter)

#     if req_pixels > total_pixels:
#         print("ERROR: Need larger file size")

#     else:
#         print(binary_chromosome[23:26])
#         if (binary_chromosome[23:26]== "000"):
#             index=0
#             index2=0
#             for p in range(total_pixels + len(delimiter) - 1):
#                 for q in range(0, 3):
#                     if index < req_pixels:
#                         array[p][q] = int(bin(array[p][q])[2:9] + binary_message[index], 2)
#                         index += 1
#                     elif (index < req_pixels + len(delimiter)):
#                         array[p][q] = int(bin(array[p][q])[2:9] + delimiter[index2], 2)
#                         index2 += 1
        

#         array=array.reshape(height, width)
#         enc_img = Image.fromarray(array.astype('uint8'), img.mode)
#         enc_img.save(dest)
        
# def Decode(src):

#     img = Image.open(src, 'r')
#     array = np.array(list(img.getdata()))
#     total_pixels = len(array)

#     hidden_bits = ""
#     for p in range(total_pixels):
#         for q in range(0, 3):
#             hidden_bits += (bin(array[p][q])[2:][-1])

#     hidden_bits = [hidden_bits[i:i+8] for i in range(0, len(hidden_bits), 8)]

#     message = ""
#     for i in range(len(hidden_bits)):
#         if message[-5:] == "##END":
#             break
#         else:
#             message += chr(int(hidden_bits[i], 2))
#     if "##END" in message:
#         return message[:-5]
#         print("Hidden Message:", message[:-5])
#     else:
#         print("No Hidden Message Found")

def hide_bit(value, bit):
    bin_pixel_str = format(value,'b')
    bin_pixel = list(bin_pixel_str)
    if(bin_pixel[len(bin_pixel) - 1] != bit):
        bin_pixel[len(bin_pixel) - 1] = bit
    bin_pixel_str = "".join(bin_pixel)
    return int(bin_pixel_str, 2)
    

def Encode(host, binary_message, binary_chromosome):
    global delimiter
    height, width = host.shape
    index = 0
    index2 = 0
    #first direction
    if (binary_chromosome[23:26]== "000"):
        for h in range(height):
            for w in range(width):
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #second direction
    elif (binary_chromosome[23:26]== "001"):
        for w in range(width):
            for h in range(height):
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #third direction
    elif (binary_chromosome[23:26]== "010"):
        for h in range(height):
            for w in range(width-1, 0, -1):
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #fourth direction
    elif (binary_chromosome[23:26]== "011"):
        for w in range(width-1, 0, -1):
            for h in range(height):
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #fifth direction
    elif (binary_chromosome[23:26]== "100"): 
         for w in range(width-1, 0, -1):
            for h in range(height-1, 0, -1):
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #sixth direction
    elif (binary_chromosome[23:26]== "101"):
        for h in range(height-1, 0, -1): 
            for w in range(width-1, 0, -1):
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #seventh direction
    elif (binary_chromosome[23:26]== "110"):
        for h in range(height-1, 0, -1): 
            for w in range(width):
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #eigth direction
    elif (binary_chromosome[23:26]== "111"):
        for w in range(width):
            for h in range(height-1, 0, -1):            
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    return host                    

def Decode(host, binary_chromosome):
    global delimiter
    height, width = host.shape
    binary_data = ""
    #first direction
    if (binary_chromosome[23:26]== "000"):
        for h in range(height):
            for w in range(width):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #second direction
    elif (binary_chromosome[23:26]== "001"):
        for w in range(width):
            for h in range(height):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #third direction
    elif (binary_chromosome[23:26]== "010"):
        for h in range(height):
            for w in range(width-1, 0, -1):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #fourth direction
    elif (binary_chromosome[23:26]== "011"):
        for w in range(width-1, 0, -1):
            for h in range(height):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #fifth direction
    elif (binary_chromosome[23:26]== "100"):
        for w in range(width-1, 0, -1):
            for h in range(height-1, 0, -1):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #sixth direction
    elif (binary_chromosome[23:26]== "101"):
        for h in range(height-1, 0, -1):
            for w in range(width-1, 0, -1):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #seventh direction
    elif (binary_chromosome[23:26]== "110"):
        for h in range(height-1, 0, -1):
            for w in range(width):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #eigth direction
    elif (binary_chromosome[23:26]== "111"):
        for w in range(width):
            for h in range(height-1, 0, -1):  
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    
    return binary_data[:-40]
# lst = list(binary_chromosome)            
# lst[23:26] = ['1','1','1']
# binary_chromosome = ''.join(lst)
# print(binary_chromosome)
image = cv2.imread('Lenna(test_image).png', 0)

#img = Image.open('Lenna(test_image).png', 'r')
img = Image.open('Secret Lenna.png', 'r')
array = np.array(list(img.getdata()))
img = img.convert('L')
binary_data, width, height = transform_secret_image(img, binary_chromosome)
print(len(binary_data), width, height)




lol = Encode(image, binary_data, binary_chromosome)
cv2.imshow('Hidden',lol)
diff = image - lol

cv2.imshow('Difference', diff)
recovered_data = Decode(lol, binary_chromosome)

if(recovered_data == binary_data):
    print('success')
    img_back = transform_bits_image(recovered_data, 100, 100, binary_chromosome)
    img_back.save('reconstructed_image.png')
    img2 = cv2.imread('reconstructed_image.png')
    cv2.imshow('Recovered', img2)   

cv2.waitKey(0)