import numpy as np
from PIL import Image
import random
import cv2

delimiter = '##END'
delimiter = ''.join(format(ord(char), '08b') for char in delimiter)
image = cv2.imread('Lenna(test_image).png', 0)
img = Image.open('Secret Lenna.png', 'r')
img = img.convert('L')

def create_chromosome():        
    chromosome = random.randint(0,2097152)
    binary_chromosome = format(chromosome,'b')
    while(len(binary_chromosome)<21):
        binary_chromosome = "0" + binary_chromosome
    return binary_chromosome


def transform_secret_image(img, binary_chromosome):
    binary_data = ''.join(format(pixel, '08b') for pixel in img.tobytes())
    if(binary_chromosome[1] == '1'):
        binary_data = ''.join('1' if bit == '0' else '0' for bit in binary_data)
    if(binary_chromosome[0] == '1'):
        binary_data = binary_data[::-1]
    return binary_data

def transform_bits_image(binary_data, width, height, binary_chromosome):
    mode = 'L'  # Grayscale mode 

    # Create an empty image
    img = Image.new(mode, (width, height))
    
    if(binary_chromosome[1] == '1'):
        binary_data = ''.join('1' if bit == '0' else '0' for bit in binary_data)
    if(binary_chromosome[0] == '1'):
        binary_data = binary_data[::-1]

    # Parse the list of bits and set pixel values
    pixels = [int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8)]
    img.putdata(pixels)
    return img


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
    if (binary_chromosome[18:21]== "000"):
        for h in range(int(binary_chromosome[2:10], 2), height):
            for w in range(int(binary_chromosome[10:18], 2), width):                       
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #second direction
    elif (binary_chromosome[18:21]== "001"):
        for w in range(int(binary_chromosome[10:18], 2), width):
            for h in range(int(binary_chromosome[2:10], 2), height):               
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #third direction
    elif (binary_chromosome[18:21]== "010"):
        for h in range(int(binary_chromosome[2:10], 2), height):
            for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):            
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #fourth direction
    elif (binary_chromosome[18:21]== "011"):
        for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
            for h in range(int(binary_chromosome[2:10], 2), height):             
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #fifth direction
    elif (binary_chromosome[18:21]== "100"): 
         for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
            for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):             
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #sixth direction
    elif (binary_chromosome[18:21]== "101"):
        for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
            for w in range(width- 1 - int(binary_chromosome[10:18], 2), 0, -1):              
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #seventh direction
    elif (binary_chromosome[18:21]== "110"):
        for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1): 
            for w in range(int(binary_chromosome[10:18], 2), width):
                if (index < len(binary_message)):
                    host[h][w] = hide_bit(host[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        host[h][w] = hide_bit(host[h][w], delimiter[index2])
                        index2 += 1
    #eigth direction
    elif (binary_chromosome[18:21]== "111"):
        for w in range(int(binary_chromosome[10:18], 2), width):
            for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):        
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
    if (binary_chromosome[18:21]== "000"):
        for h in range(int(binary_chromosome[2:10], 2), height):
            for w in range(int(binary_chromosome[10:18], 2), width): 
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #second direction
    elif (binary_chromosome[18:21]== "001"):
        for w in range(int(binary_chromosome[10:18], 2), width):
            for h in range(int(binary_chromosome[2:10], 2), height):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #third direction
    elif (binary_chromosome[18:21]== "010"):
        for h in range(int(binary_chromosome[2:10], 2), height):
            for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #fourth direction
    elif (binary_chromosome[18:21]== "011"):
        for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
            for h in range(int(binary_chromosome[2:10], 2), height):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #fifth direction
    elif (binary_chromosome[18:21]== "100"):
        for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
            for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #sixth direction
    elif (binary_chromosome[18:21]== "101"):
        for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
            for w in range(width- 1 - int(binary_chromosome[10:18], 2), 0, -1):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #seventh direction
    elif (binary_chromosome[18:21]== "110"):
        for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1): 
            for w in range(int(binary_chromosome[10:18], 2), width):
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    #eigth direction
    elif (binary_chromosome[18:21]== "111"):
        for w in range(int(binary_chromosome[10:18], 2), width):
            for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):  
                if(binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w],'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    
    return binary_data[:-40]

#TEST

# binary_chromosome = create_chromosome()
# image = cv2.imread('Lenna(test_image).png', 0)
# #img = Image.open('Lenna(test_image).png', 'r')
# img = Image.open('Secret Lenna.png', 'r')
# array = np.array(list(img.getdata()))
# img = img.convert('L')
# binary_data = transform_secret_image(img, binary_chromosome)

# lol = Encode(image, binary_data, binary_chromosome)
# cv2.imshow('Hidden',lol)
# diff = image - lol
# print(cv2.PSNR(image, image))
# recovered_data = Decode(lol,  binary_chromosome)

# img_back = transform_bits_image(recovered_data, 100, 100, binary_chromosome)
# img_back.save('reconstructed_image.png')
# img2 = cv2.imread('reconstructed_image.png')
# cv2.imshow('Recovered', img2) 



def fitness_function(binary_chromosome):
    global image, img
    binary_data = transform_secret_image(img, binary_chromosome)
    lol = Encode(image, binary_data, binary_chromosome)
    return cv2.PSNR(image, lol)
      
# Genetic Algorithm parameters
population_size = 50
num_generations = 10
mutation_rate = 0.1

# Initialization: Create an initial population
population = [create_chromosome() for _ in range(population_size)]

for generation in range(num_generations):
    # Fitness Evaluation
    fitness_scores = [fitness_function(x) for x in population]
    
    # Selection: Roulette wheel selection
    selected_parents = random.choices(population, weights=fitness_scores, k=population_size)
    
    # Crossover: One-point crossover
    offspring = []
    for i in range(0, population_size, 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1]
        crossover_point = random.randint(1, 21 - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.extend([offspring1, offspring2])

    # Mutation: Flip individual bits with a low probability
    for i in range(population_size):
        for j in range(21):
            if random.random() < mutation_rate:
                offspring[i] = offspring[i][:j] + ("0" if offspring[i][j] == "1" else "1") + offspring[i][j+1:]

    # Replacement: Replace the old population with the offspring
    population = offspring

# Print the best solution found
best_solution = max(population, key=fitness_function)
print("Best solution:", best_solution, "PSNR : ", fitness_function(best_solution))
binary_data = transform_secret_image(img, best_solution)
lol = Encode(image, binary_data, best_solution)
cv2.imshow('Hidden',lol)
diff = image - lol

cv2.imshow('Difference', diff)
recovered_data = Decode(lol, best_solution)

if(recovered_data == binary_data):
    print('success')
    img_back = transform_bits_image(recovered_data, 100, 100, best_solution)
    img_back.save('reconstructed_image.png')
    img2 = cv2.imread('reconstructed_image.png')
    cv2.imshow('Recovered', img2)
cv2.waitKey(0) 
