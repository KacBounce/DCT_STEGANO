import numpy as np
from PIL import Image
import random
import cv2
from threading import Thread, Lock

delimiter = '##END'
delimiter = ''.join(format(ord(char), '08b') for char in delimiter)
image = cv2.imread('lenna(test_image).png', 0)
genetic_lock = Lock()
pso_lock = Lock()

best_particle = ''
best_solution = ''



secret_image = Image.open('Secret Lenna.png', 'r')
secret_width, secret_height = secret_image.size
secret_image = secret_image.convert('L')

def create_chromosome():        
    chromosome = random.randint(0,2097152)
    binary_chromosome = format(chromosome,'b')
    while(len(binary_chromosome)<21):
        binary_chromosome = "0" + binary_chromosome
    return binary_chromosome


def transform_secret_image(secret_image, binary_chromosome):
    binary_data = ''.join(format(pixel, '08b') for pixel in secret_image.tobytes())
    if(binary_chromosome[1] == '1'):
        binary_data = ''.join('1' if bit == '0' else '0' for bit in binary_data)
    if(binary_chromosome[0] == '1'):
        binary_data = binary_data[::-1]
    return binary_data

def transform_bits_image(binary_data, width, height, binary_chromosome):
    mode = 'L'  # Grayscale mode 

    # Create an empty image
    secret_image = Image.new(mode, (width, height))
    
    if(binary_chromosome[1] == '1'):
        binary_data = ''.join('1' if bit == '0' else '0' for bit in binary_data)
    if(binary_chromosome[0] == '1'):
        binary_data = binary_data[::-1]

    # Parse the list of bits and set pixel values
    pixels = [int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8)]
    secret_image.putdata(pixels)
    return secret_image


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
    transformed = host.copy()
    index = 0
    index2 = 0
    #first direction
    if (binary_chromosome[18:21]== "000"):
        for h in range(int(binary_chromosome[2:10], 2), height):
            for w in range(int(binary_chromosome[10:18], 2), width):                       
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(transformed[h][w], delimiter[index2])
                        index2 += 1
    #second direction
    elif (binary_chromosome[18:21]== "001"):
        for w in range(int(binary_chromosome[10:18], 2), width):
            for h in range(int(binary_chromosome[2:10], 2), height):               
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(transformed[h][w], delimiter[index2])
                        index2 += 1
    #third direction
    elif (binary_chromosome[18:21]== "010"):
        for h in range(int(binary_chromosome[2:10], 2), height):
            for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):            
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(transformed[h][w], delimiter[index2])
                        index2 += 1
    #fourth direction
    elif (binary_chromosome[18:21]== "011"):
        for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
            for h in range(int(binary_chromosome[2:10], 2), height):             
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(transformed[h][w], delimiter[index2])
                        index2 += 1
    #fifth direction
    elif (binary_chromosome[18:21]== "100"): 
         for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
            for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):             
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(transformed[h][w], delimiter[index2])
                        index2 += 1
    #sixth direction
    elif (binary_chromosome[18:21]== "101"):
        for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
            for w in range(width- 1 - int(binary_chromosome[10:18], 2), 0, -1):              
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(transformed[h][w], delimiter[index2])
                        index2 += 1
    #seventh direction
    elif (binary_chromosome[18:21]== "110"):
        for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1): 
            for w in range(int(binary_chromosome[10:18], 2), width):
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(transformed[h][w], delimiter[index2])
                        index2 += 1
    #eigth direction
    elif (binary_chromosome[18:21]== "111"):
        for w in range(int(binary_chromosome[10:18], 2), width):
            for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):        
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if(index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(transformed[h][w], delimiter[index2])
                        index2 += 1
    return transformed                    

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
# #secret_image = Image.open('Lenna(test_image).png', 'r')
# secret_image = Image.open('Secret Lenna.png', 'r')
# array = np.array(list(secret_image.getdata()))
# secret_image = secret_image.convert('L')
# binary_data = transform_secret_image(secret_image, binary_chromosome)

# lol = Encode(image, binary_data, binary_chromosome)
# cv2.imshow('Hidden',lol)
# diff = image - lol
# print(cv2.PSNR(image, image))
# recovered_data = Decode(lol,  binary_chromosome)

# secret_image_back = transform_bits_image(recovered_data, 100, 100, binary_chromosome)
# secret_image_back.save('reconstructed_image.png')
# secret_image2 = cv2.imread('reconstructed_image.png')
# cv2.imshow('Recovered', secret_image2) 


def fitness_function(binary_chromosome):
    global image, secret_image
    binary_data = transform_secret_image(secret_image, binary_chromosome)
    secret = Encode(image, binary_data, binary_chromosome)
    return cv2.PSNR(image, secret) 


def genetic():      
    global best_solution
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
    genetic_lock.acquire()
    # Print the best solution found
    best_solution = max(population, key=fitness_function)
    genetic_lock.release()

def pso():
    global secret_image, best_particle
    # Define the problem parameters
    num_particles = 30
    num_dimensions = 21  # Length of the binary chromosome
    max_iterations = 100
    c1 = 2.0  # Cognitive coefficient
    c2 = 2.0  # Social coefficient
    # Initialize the particles' positions and velocities
    global_best_position = None
    best_positions = []  # Best positions for each particle
    
    particles = []
    for _ in range(num_particles):
        particle = [random.choice([0, 1]) for _ in range(num_dimensions)]
        particles.append(particle)
        best_positions.append(list(particle))
        
    velocities = []
    for _ in range(num_particles):
        velocity = [random.uniform(-1, 1) for _ in range(num_dimensions)]
        velocities.append(velocity)
    # PSO Loop
    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Evaluate the fitness of the current particle
            fitness = fitness_function(particles[i])

            # Update the best-known position for the particle
            if fitness_function(best_positions[i]) < fitness:
                best_positions[i] = list(particles[i])

            # Update the global best position
            if global_best_position is None or fitness_function(global_best_position) < fitness:
                global_best_position = list(particles[i])

            # Update the particle's velocity and position
            for j in range(num_dimensions):
                r1, r2 = random.random(), random.random()
                cognitive_component = c1 * r1 * (best_positions[i][j] - particles[i][j])
                social_component = c2 * r2 * (global_best_position[j] - particles[i][j])
                velocities[i][j] = velocities[i][j] + cognitive_component + social_component
                # Update the particle's position (using binary update)
                particles[i][j] = 1 if random.random() < 1 / (1 + 2.7**(-velocities[i][j])) else 0
    pso_lock.acquire()
    best_particle = ["1" if x == 1 else "0" for x in global_best_position]
    best_particle = ''.join(best_particle)
    pso_lock.release()

def multithreaded_optimization():
    threads_genetic = []
    threads_pso = []
    num_threads = 4
    for i in range(num_threads):
        x = Thread(target=genetic)
        y = Thread(target=pso)
        threads_genetic.append(x)
        threads_pso.append(y)
    for i in range(num_threads):
        threads_pso[i].start()
        print("Starting a pso thread")
        threads_genetic[i].start()
        print("Starting a genetic thread")
    print("Working on the optimization...")
    for i in range(num_threads):
        threads_pso[i].join()
        print("Stopping a pso thread")
        threads_genetic[i].join()
        print("Stopping a genetic thread")


def main():
    multithreaded_optimization()
    
    print("Best solution from Genetic", best_solution, "PSNR : ", fitness_function(best_solution))
    binary_data = transform_secret_image(secret_image, best_solution)
    print(len(binary_data))
    lol = Encode(image, binary_data, best_solution)
    cv2.imshow('Hidden genetic',lol)
    cv2.imwrite('Hidden_lsb_genetic.png', lol)
    diff = image - lol

    cv2.imshow('Difference genetic', diff)
    recovered_data = Decode(lol, best_solution)

    if(recovered_data == binary_data):
        print('success genetic')
        secret_image_back = transform_bits_image(recovered_data, secret_width, secret_height, best_solution)
        secret_image_back.save('reconstructed_image_genetic.png')
        secret_image2 = cv2.imread('reconstructed_image_genetic.png')
        cv2.imshow('Recovered genetic', secret_image2)
    else:
        print('Failed genetic')
    
    print("Best solution pso :", best_particle, "PSNR : ", fitness_function(best_particle))
    binary_data = transform_secret_image(secret_image, best_particle)
    lol = Encode(image, binary_data, best_particle)
    cv2.imshow('Hidden pso',lol)
    cv2.imwrite('Hidden_lsb_pso.png', lol)
    diff = image - lol

    cv2.imshow('Difference pso', diff)
    recovered_data = Decode(lol, best_particle)

    if(recovered_data == binary_data):
        print('success pso')
        secret_image_back = transform_bits_image(recovered_data, secret_width, secret_height, best_particle)
        secret_image_back.save('reconstructed_image_pso.png')
        secret_image2 = cv2.imread('reconstructed_image_pso.png')
        cv2.imshow('Recovered_pso', secret_image2)
    else:
        print('Failed pso')
    cv2.waitKey(0) 

main()

        