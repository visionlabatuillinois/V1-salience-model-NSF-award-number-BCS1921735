import os
import sys
import gc
import trig
import math
from datetime import datetime, timedelta
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing as mp
#from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


#from simple_tools import *
from constants_4 import *


# Setup paths and files to run
# TODO we may want to pass the file in as an argument
#files_to_run = ['small_dragon.png']


#x = [i[2] for i in os.walk(inpath)]
#files_to_run=['i1007068829.jpeg']

cpu_count = os.cpu_count() - 1 # Need rank 0 to manage the pool

# Create locations
def construct_hex_locations():

    hypercolumn_locations = []
    print("Begin construct_hex_locations" + str(datetime.now()))
    vf_center = [VISUAL_FIELD_RADIUS, VISUAL_FIELD_RADIUS]


    for rf_radius in V1_HYPERCOLUMN_SCALES:
        print("DEBUG rf_radius :" + str(rf_radius))
        rf_radius_index = V1_HYPERCOLUMN_SCALES.index(rf_radius)
        hypercolumn_locations.append([])
        vf_ring_index = 0
        vf_ring_radius = 0

        current_rf_radius = float(rf_radius)
        # This is the location at the center of the visual field
        # Append a list containing scale index in scale list, rf_radius in px, the polar coordinates in the visual field, and cartesian center
        hypercolumn_locations[-1].append(
            [rf_radius_index, current_rf_radius, 0, 0, [VISUAL_FIELD_RADIUS, VISUAL_FIELD_RADIUS]])

        while (vf_ring_radius + current_rf_radius < VISUAL_FIELD_RADIUS):
            vf_ring_index += 1
            spacing = V1_HYPERCOLUMN_OFFSET * rf_radius + vf_ring_index * CORTICAL_MAGNIFICATION * rf_radius
            vf_ring_radius += spacing
            vf_angle = 0.0

            current_rf_radius = rf_radius * math.pow((1.0 + CORTICAL_MAGNIFICATION), vf_ring_index)

            # Create the rays extending every 60 degrees
            iso_points = []
            while vf_angle <= (5 * math.pi) / 3:  # while a full circle has not been traversed
                # make a new hypercolumn location at this vf radius and vf angle
                # Calculate the center in cartesian coordinates
                vf_x = math.cos(vf_angle) * vf_ring_radius
                vf_y = math.sin(vf_angle) * vf_ring_radius

                # now get screen coordinates
                int_x = vf_center[0] + int(round(vf_x))
                int_y = vf_center[1] + int(round(vf_y))  # screen_center[1]-int(round(vf_y))
                center = [int_x, int_y]
                iso_points.append(
                    [rf_radius_index, int(round(current_rf_radius)), vf_ring_radius, vf_angle, center])
                vf_angle += math.pi / 3.0  # 60 degrees in radians


            for point_index in range(len(iso_points)):  # Now interpolate between the rays to add other lattice points
                in_range_iso_points  = []
                interpolation_points = []
                # The last element in the list for a given location is the cartesian center
                (x1, y1) = iso_points[point_index][4]
                (x2, y2) = iso_points[(point_index + 1) % 6][4]
                # Some of the iso points calculated previously might be outside of the visual field boundary
                if x1 > 0 and x1 < 2*VISUAL_FIELD_RADIUS and y1 > 0 and y1 < 2*VISUAL_FIELD_RADIUS:
                    in_range_iso_points.append(iso_points[point_index])
                weight = 0
                num_points = vf_ring_index - 1
                while len(interpolation_points) < num_points:
                    # there are always one less than the ring number of points to interpolate
                    weight += 1.0 / (1 + num_points)
                    opposite_weight = 1.0 - weight
                    new_point = [(x1 * weight + x2 * opposite_weight), (y1 * weight + y2 * opposite_weight)]
                    new_point[0] = int(round(new_point[0]))
                    new_point[1] = int(round(new_point[1]))
                    (r, theta) = trig.get_polar(new_point, [0, 0])
                    if weight > 0:
                        if new_point[0] > 0 and new_point[0] < 2*VISUAL_FIELD_RADIUS and new_point[1] > 0 and new_point[1] < 2*VISUAL_FIELD_RADIUS:
                            interpolation_points.append([rf_radius_index, int(round(current_rf_radius)), r, theta, new_point])

                hypercolumn_locations[-1].extend(in_range_iso_points)
                hypercolumn_locations[-1].extend(interpolation_points)
    return(hypercolumn_locations)


def generate_2dgaussian(radius, sigma, mu=0.0):
    x, y = np.meshgrid(np.linspace(-1, 1, 2*radius), np.linspace(-1, 1, 2*radius))
    # calculate the distance from the center at each point
    dst = np.sqrt(x*x + y*y)
    g = (np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))))/(math.sqrt(2.0*math.pi)*sigma)
    return g

def generate_filters(radius):
    filter_deck = []
    x, y = np.meshgrid(np.linspace(-1, 1, 2 * radius), np.linspace(-1, 1, 2 * radius))
    gaussian = generate_2dgaussian(radius, FILTER_SIGMA)
    point_angle = np.zeros((radius*2, radius*2))
    dst = np.sqrt(x * x + y * y)
    for xi in range(len(x)):
        for yj in range(len(y)):

            point = [xi, yj]
            center = [radius, radius]
            distance, angle = trig.get_polar(point, center)
            point_angle[xi, yj] = angle

    for phase in PHASES:
        for frequency in FREQUENCIES:

            for orn_index in range(NUM_ORIENTATIONS):
                orn = orn_index*math.pi/8.0
                functional_orn =  orn - math.pi/2.0
                if functional_orn < 0.0:
                    functional_orn += 2 * math.pi
                point_projection = np.cos(point_angle - functional_orn) * dst * frequency


                # Add a filter for each

                if phase == 'sin':
                    filter = np.sin(point_projection)*gaussian
                elif phase == 'cos':
                    filter = np.cos(point_projection)*gaussian
                filter_integral = np.sum(filter)
                filter_a_correction = filter_integral/filter.size
                filter -= filter_a_correction


                # Apply a multiplicative correction that is a ratio of the smallest RF area to the current RF area
                smallest_rf_area = (V1_HYPERCOLUMN_SCALES[0]*2)**2
                filter_area = filter.size
                filter_m_correction = 10*smallest_rf_area/filter_area
                filter *= filter_m_correction

                filter_deck.append([orn_index, phase, frequency, filter])
    return filter_deck


def get_image(file_path):
    '''
    read the image file that you'll use.
    this separated from V1 init on 3/10/19 in response to errors with saving naked (locations only) V1
    :param file_name:
    :return:
    '''
    file_name = file_path # give it the new filename

    raw_image     = Image.open(file_name)         #  PIL.Image, which has other PIL stuff associated with it
    image_width, image_height = raw_image.size                                 #   Size of the image

    # construct a "padded" image, so that the raw image, which can be any size, is shoe-horned into an array
    # of dimension 2*vf_radius X 2*vf_radius. At each location in this array is a tuple (R,G,B)
    # the padded image will be white everywhere the raw_image isn't
    padded = Image.new('RGB',
             (VISUAL_FIELD_RADIUS * 2, VISUAL_FIELD_RADIUS * 2),   # A4 at 72dpi
             (255, 255, 255))  # White

    # shoehorn the raw image into the center of the padded array
    padded.paste(raw_image, ((int(math.floor(VISUAL_FIELD_RADIUS*2 - image_width)/2)), math.floor(int((VISUAL_FIELD_RADIUS*2 - image_height)/2))))  # centered
    numpy_image = np.array(padded)

    return image_width, image_height, numpy_image



def generate_salience_map(outpath, filename, image_width, image_height, hypercolumn_locations):
    max_salience_value = 0
    #print("location shape " + str(hypercolumn_locations.shape))
    print("Image width " + str(image_width))
    print("Image height " + str(image_height))
    total_activity = np.zeros(1)
    salience_map = np.zeros((VISUAL_FIELD_RADIUS * 2, VISUAL_FIELD_RADIUS * 2), dtype=int)
    for scale in hypercolumn_locations:

        for location in scale:
            #print(location)

            hypercolumn_filter_sum = 0

            if len(location) == 6:
                #print("LOCATION " + str(location))

                #print(location)
                for neuron in location[5]: # this is the list of neuron attributes
                    #print(neuron[4])
                    hypercolumn_filter_sum += abs(neuron[4]) # This is the contrast salience of the whole hypercolumn
                    #print("NEURON " + str(neuron))
            #print("FILTER SUM " + str(hypercolumn_filter_sum))

            center = location[4] # the fourth element is the cartesian center
            rf_radius = int(round(location[1]))
            # These are receptive field dimensions for the location in question
            min_pixel_x = int(round(center[0] - rf_radius))
            max_pixel_x = int(round(center[0] + rf_radius))
            min_pixel_y = int(round(center[1] - rf_radius))
            max_pixel_y = int(round(center[1] + rf_radius))

            # if min_pixel_x < math.ceil((VISUAL_FIELD_RADIUS * 2 - image_width) / 2): min_pixel_x = int(
            #     math.ceil((VISUAL_FIELD_RADIUS * 2 - image_width) / 2))
            # if min_pixel_y < math.ceil((VISUAL_FIELD_RADIUS * 2 - image_height) / 2): min_pixel_y = int(
            #     math.ceil((VISUAL_FIELD_RADIUS * 2 - image_height) / 2))n
            # if max_pixel_x > math.floor((VISUAL_FIELD_RADIUS * 2 + image_width) / 2): max_pixel_x = int(
            #     math.floor((VISUAL_FIELD_RADIUS * 2 + image_width) / 2))
            # if max_pixel_y > math.floor((VISUAL_FIELD_RADIUS * 2 + image_height) / 2): max_pixel_y = int(
            #     math.floor((VISUAL_FIELD_RADIUS * 2 + image_height) / 2))
            # This is where the image starts and stops within the visual field
            x_min_img_bound = math.ceil((VISUAL_FIELD_RADIUS * 2 - image_width) / 2)
            y_min_img_bound = math.ceil((VISUAL_FIELD_RADIUS * 2 - image_height) / 2)
            x_max_img_bound = math.floor((VISUAL_FIELD_RADIUS * 2 + image_width) / 2)
            y_max_img_bound = math.floor((VISUAL_FIELD_RADIUS * 2 + image_height) / 2)
            if min_pixel_x > x_min_img_bound and min_pixel_y > y_min_img_bound and max_pixel_x < x_max_img_bound and max_pixel_y < y_max_img_bound:
                #blur_gaussian = generate_2dgaussian(rf_radius, FILTER_SIGMA)
                #delta = blur_gaussian*hypercolumn_filter_sum

            # TODO THE LOCATIONS ARE THERE! But there are NO NEURONS IN THOSE LOCATIONS!  WHY??
                #TODO well here is the problem. The radius should probably be constant fo the salience map
                blur_gaussian = generate_2dgaussian(rf_radius, FILTER_SIGMA)
                # Apply a multiplicative correction that is a ratio of the smallest RF area to the current RF area
                smallest_rf_area = (V1_HYPERCOLUMN_SCALES[0]*2)**2
                blur_area = blur_gaussian.size
                m_correction = 10*smallest_rf_area/blur_area

                blur_gaussian *= m_correction



                #delta = blur_gaussian*1.0 # show locations
                delta = blur_gaussian * hypercolumn_filter_sum
                total_activity += np.sum(delta)
                #print(delta.shape)
                salience_map[min_pixel_y:max_pixel_y, min_pixel_x:max_pixel_x] =  salience_map[min_pixel_y:max_pixel_y,min_pixel_x:max_pixel_x] + delta

    #plt.imshow(activity_map)
    #plt.show()

    print(total_activity)
    if total_activity > 0.0: salience_pdf = salience_map/total_activity
    else: salience_pdf = np.zeros((VISUAL_FIELD_RADIUS * 2, VISUAL_FIELD_RADIUS * 2))
    #plt.imshow(salience_pdf)
    #plt.show()
    pdfsum = np.sum(salience_pdf)
    pdfmax = np.max(salience_pdf)
    print(pdfsum)
    salience_gray_values = np.round(salience_pdf*255.0/pdfmax).astype(np.uint8)
    #print(salience_gray_values)
    salience_map_image = []
    for x in range(len(salience_gray_values)):
        for y in range(len(salience_gray_values[x])):
            salience_map_image.append((salience_gray_values[x][y], salience_gray_values[x][y],salience_gray_values[x][y]))

    #salience_map_image = np.dstack([salience_gray_values]*3)
    #print(salience_map_image.shape)
    #np.reshape(salience_map_image, (VISUAL_FIELD_RADIUS * 2, VISUAL_FIELD_RADIUS* 2, 3))
    #salience_map_image = [tuple(salience_map_image[row][column]) for row in salience_map_image for column in row]
    #print(salience_map_image.shape)
    image_square = Image.new('RGB', (VISUAL_FIELD_RADIUS * 2, VISUAL_FIELD_RADIUS* 2), (0, 0, 0))  # Black
    left = 0
    right = VISUAL_FIELD_RADIUS * 2
    upper = 0
    lower = VISUAL_FIELD_RADIUS * 2
    image_square.putdata(salience_map_image)

    if (image_width < 2 * VISUAL_FIELD_RADIUS):
        left = VISUAL_FIELD_RADIUS - image_width / 2
        right = 2 * VISUAL_FIELD_RADIUS - (2 * VISUAL_FIELD_RADIUS - image_width) / 2
    if (image_height < 2 * VISUAL_FIELD_RADIUS):
        upper = VISUAL_FIELD_RADIUS - image_height / 2
        lower = 2 * VISUAL_FIELD_RADIUS - (2 * VISUAL_FIELD_RADIUS - image_height) / 2

    print("("+str(left) + ", " + str(upper) + ", " + str(right) + ", " + str(lower))

    image_out = image_square.crop((left, upper, right, lower))
    save_filename = str(outpath) + str(filename)
    print(filename)
    print(save_filename)
    image_out.save(str(save_filename))
    return(image_out)


def run_filters_for_scale(input_tuple):
    print("Running filters at " + str(datetime.now()))
    location_list, image, image_width, image_height = input_tuple
    # Calculate where the image itself ends
    x_min_img_bound = math.ceil((VISUAL_FIELD_RADIUS - image_width / 2))
    y_min_img_bound = math.ceil((VISUAL_FIELD_RADIUS - image_height / 2))
    x_max_img_bound = math.floor((VISUAL_FIELD_RADIUS + image_width / 2))
    y_max_img_bound = math.floor((VISUAL_FIELD_RADIUS + image_height / 2))


    for location in location_list:
        neuron_data = []
        WB_convolve = 0.0
        RG_convolve = 0.0
        BY_convolve = 0.0
        radius = int(round(location[1]))
        center = location[4]

        filter_deck = generate_filters(radius)

        min_pixel_x = int(round(center[0] - radius))  # TODO should these be floor or ceiling instad?
        max_pixel_x = int(round(center[0] + radius))
        min_pixel_y = int(round(center[1] - radius))
        max_pixel_y = int(round(center[1] + radius))


        if min_pixel_x >= x_min_img_bound and min_pixel_y >= y_min_img_bound and max_pixel_x <= x_max_img_bound and max_pixel_y <= y_max_img_bound:

            red_patch = image[min_pixel_y:max_pixel_y, min_pixel_x:max_pixel_x, 0]
            green_patch = image[min_pixel_y:max_pixel_y, min_pixel_x:max_pixel_x, 1]
            blue_patch = image[min_pixel_y:max_pixel_y, min_pixel_x:max_pixel_x, 2]
            # WB is filter_value * (R + G + B - 0)/(255 * 3)

            for filter in filter_deck:
                WB_product = filter[3] * ((red_patch + green_patch + blue_patch)) / 765.0
                WB_convolve += np.sum(WB_product)
                RG_product = filter[3] * ((red_patch - green_patch)) / 255.0
                RG_convolve += np.sum(RG_product)
                BY_product = filter[3] * ((2 * blue_patch - (red_patch + green_patch))) / 510.0
                BY_convolve += np.sum(BY_product)
                # Now we have to add the data that corresponds to neurons (create the neurons later)
                sums = [WB_convolve, RG_convolve, BY_convolve]
                for sum_index in range(len(sums)):
                    if abs(sums[sum_index]) > V1_MIN_FILTER_THRESHOLD:
                        neuron_data.append([filter[0], filter[1], filter[2], sum_index, sums[sum_index]])
        location.append(neuron_data)

    return location_list

def rbind(comm, x):
    return np.vstack(comm.allgather(x))

def main():
    ranks = int(sys.argv[1]) - 1
    #inpath = sys.argv[2] #'./data/MIT1003/ALLSTIMULI/'
    #outpath = sys.argv[3] #'./data/MIT1003/CASPER_MIT1003_OUTPUT_CONTRAST/'
    inpath = './data/MIT1003/ALLSTIMULI/'
    outpath = './data/completed_runs/MIT1003/CASPER_MIT1003_CONTRAST_16082021_2/'

    x = [i[2] for i in os.walk(inpath)]
    files_to_run = []
    for t in x:
        for f in t:
            files_to_run.append(f)


    for file in files_to_run:
        filename = file
        print("Beginning processing of " + str(filename) + " at " + str(datetime.now()))
        if os.path.exists(outpath + os.path.splitext(filename)[0] + '.jpeg'):
            print("Contrast map already exists")
        else:
            location_scales = construct_hex_locations()
            file_path = inpath+filename
            image_width, image_height, image = get_image(file_path)

            updated_scales = []
            for scale in location_scales:
                print("Running scale " + str(location_scales.index(scale)) + " at " + str(datetime.now()))
                scale_chunks = list([scale[int(len(scale) * i / (ranks * 2)):int(len(scale) * (i + 1) / (ranks * 2))] for i in range(ranks * 2)])
                arglist = [(scale_chunk, image, image_width, image_height) for scale_chunk in scale_chunks]


                executor = MPIPoolExecutor()
                updated_chunked_scale = list(executor.map(run_filters_for_scale, arglist))
                executor.shutdown()

                updated_scale =[]
                for chunk in updated_chunked_scale:
                    updated_scale.extend(chunk)
                updated_scales.append(updated_scale)

            generate_salience_map(outpath, filename, image_width, image_height, updated_scales)

if __name__ == "__main__":
    main()
