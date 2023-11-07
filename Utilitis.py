import math
from statistics import mean

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, patches
import FeaturesCalculation
from PIL import Image
from matplotlib.patches import Polygon

"""
    we ask the user of the script to assign the dominant hand of the signer.   
    The function take what the user assigned and translate it to the matching column name 
    from data_per_frame

    @param DominantHand: what the user assigned. string.
    @return dominantHand_col_name: string. the name of the matching column from data_per_frame.
"""


def donminant_hand_filter(DominantHand):
    if DominantHand == 'right' or DominantHand == 'Right' or DominantHand == 'HandRight' or DominantHand == 'R':
        dominantHand_col_name = 'HandRight'
    elif DominantHand == 'left' or DominantHand == 'Left' or DominantHand == 'HandLeft' or DominantHand == 'L':
        dominantHand_col_name = 'HandLeft'
    else:
        print('DominantHand is not well defined. Please fill in "right" or "left"')
        exit()
    return dominantHand_col_name


"""
    The function calculates the coordinates of the average body coordinate of the first 2 frames.

    @param df: data_per_frame
    @return Body1, Body2, Body3: arrays. the mean coordinates of SpineNavel, ClavicleRight, ClavicleLeft
                                 of the first 2 frames.
"""


def average_body_coordinates_calculation(df):
    # Body coordinates of first 2 frames
    first_SpineNavel_vec = np.array(df.loc[0, 'SpineNavel'])
    first_ClavicleRight_vec = np.array(df.loc[0, 'ClavicleRight'])
    first_ClavicleLeft_vec = np.array(df.loc[0, 'ClavicleLeft'])
    second_SpineNavel_vec = np.array(df.loc[1, 'SpineNavel'])
    second_ClavicleRight_vec = np.array(df.loc[1, 'ClavicleRight'])
    second_ClavicleLeft_vec = np.array(df.loc[1, 'ClavicleLeft'])

    # Average body coordinates of first 2 frames
    mean_SpineNavel = FeaturesCalculation.average_position_calculation([first_SpineNavel_vec, second_SpineNavel_vec])
    mean_ClavicleRight = FeaturesCalculation.average_position_calculation(
        [first_ClavicleRight_vec, second_ClavicleRight_vec])
    mean_ClavicleLeft = FeaturesCalculation.average_position_calculation(
        [first_ClavicleLeft_vec, second_ClavicleLeft_vec])

    # Convert average body coordinates of first 2 frames arrays
    Body1 = np.array(mean_SpineNavel)
    Body2 = np.array(mean_ClavicleRight)
    Body3 = np.array(mean_ClavicleLeft)

    return Body1, Body2, Body3


"""
    The function calculates the average coordinate of right and left eye 

    @param df: data_per_frame df
    @param columns: list of the right and left eyes column names.

    @return middle_of_the_eyes_column: list. a column contains the mean of the right and left columns.
"""


def middle_of_the_eyes_calculation(df, columns):
    middle_of_the_eyes_column = []
    for _, row in df.iterrows():
        left_eye_coord = row[columns[0]]
        right_eye_coord = row[columns[1]]

        coord1 = np.array(left_eye_coord)
        coord2 = np.array(right_eye_coord)

        midpoint = np.mean([coord1, coord2], axis=0)
        middle_of_the_eyes_column.append(midpoint.tolist())  # Convert midpoint to list and append to the result list

    return middle_of_the_eyes_column


"""
    set the connections between joins (the lines connecting joints on skeletons)

    @return joints: list of connections
"""


def set_connections():
    connections = [
        ('Pelvis', 'SpineNavel'),
        ('SpineNavel', 'SpineChest'),
        ('SpineChest', 'Neck'),
        ('SpineChest', 'ClavicleLeft'),
        ('ClavicleLeft', 'ShoulderLeft'),
        ('ShoulderLeft', 'ElbowLeft'),
        ('ElbowLeft', 'WristLeft'),
        ('WristLeft', 'HandLeft'),
        ('HandLeft', 'HandTipLeft'),
        ('HandLeft', 'ThumbLeft'),
        ('SpineChest', 'ClavicleRight'),
        ('ClavicleRight', 'ShoulderRight'),
        ('ShoulderRight', 'ElbowRight'),
        ('ElbowRight', 'WristRight'),
        ('WristRight', 'HandRight'),
        ('HandRight', 'HandTipRight'),
        ('HandRight', 'ThumbRight'),
        ('Pelvis', 'HipLeft'),
        ('HipLeft', 'KneeLeft'),
        ('KneeLeft', 'AnkleLeft'),
        ('AnkleLeft', 'FootLeft'),
        ('Pelvis', 'HipRight'),
        ('HipRight', 'KneeRight'),
        ('KneeRight', 'AnkleRight'),
        ('AnkleRight', 'FootRight'),
        ('Neck', 'Head'),

    ]
    return connections


"""
    set th joints want to be drawn on skeleton images

    @return joints: list of joints
"""


def set_joints():
    joints = [
        'Pelvis',
        'SpineNavel',
        'SpineChest',
        'Neck',
        'ClavicleLeft',
        'ShoulderLeft',
        'ElbowLeft',
        'WristLeft',
        'HandLeft',
        'HandTipLeft',
        'ThumbLeft',
        'ClavicleRight',
        'ShoulderRight',
        'ElbowRight',
        'WristRight',
        'HandRight',
        'HandTipRight',
        'ThumbRight',
        'HipLeft',
        'KneeLeft',
        'AnkleLeft',
        'FootLeft',
        'HipRight',
        'KneeRight',
        'AnkleRight',
        'FootRight',
        'Head'
    ]
    return joints


"""
    The function converts the coordinates in df from string to float so we would be able to use them for calculations.

    @param df: the data per frame xlsx
    @type df: pandas df

    @return df: the same df but with float coordinates instead of string.
"""


def convert_coordinates_to_float(df):
    coordinates_start_idx = df.columns.get_loc('Pelvis')
    for idx in range(coordinates_start_idx, len(df.columns), 2):
        coordinate_column = df.iloc[:, idx]
        new_coordinate_column = []
        for coordinate in coordinate_column:
            split_coordinate = coordinate.split(',')
            x = float(split_coordinate[0])
            y = float(split_coordinate[1])
            z = float(split_coordinate[2])
            new_coordinate = [x, y, z]
            new_coordinate_column.append(new_coordinate)
        df.iloc[:, idx] = pd.Series(new_coordinate_column)
    return df


def convert_one_coordinate_to_float(coordinate):
    split_coordinate = coordinate.split('[')
    split_coordinate = coordinate.split(',')
    x = float(split_coordinate[0])
    y = float(split_coordinate[1])
    z = float(split_coordinate[2])
    new_coordinate = [x, y, z]
    return new_coordinate


"""
    The function extract the coordinates of a given joint for a specific frame

    @param df: the data per frame xlsx
    @type df: pandas df
    @param file_name: name of the frame/image
    @type file_name: string
    @param joints: chosen joints
    @type joints: list

    @return coordinates_dict: name of joint as key, coordinates as values
"""


def extract_skeleton_coordinates(df, file_name, joints):
    row = df[df['File Name'] == file_name]
    coordinates_dict = {}
    for joint in joints:
        coordinates_dict[joint] = row[joint].values[0]

    return coordinates_dict


def extract_word_name_from_directory_path(root):
    split_root = root.split('\\')
    split_folder_name = split_root[1].split('_')
    word_name_vector = split_folder_name[5:-1]
    word_name = '_'.join(word_name_vector)
    return word_name

def extract_dominant_hand_name_from_directory_path(root):
    split_root = root.split('\\')
    split_folder_name = split_root[1].split('_')
    dominant_hand_vector = split_folder_name[-1]
    dominant_hand = '_'.join(dominant_hand_vector)
    return dominant_hand


def draw_skeleton(coordinates_dict, connections, old_coordinates):
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the joints
    for joint in coordinates_dict:
        x = coordinates_dict[joint][0]
        y = coordinates_dict[joint][1]
        z = coordinates_dict[joint][2]

        ax.plot(x, -y, 'ro')  # 'ro' represents red circles for joints, y-coordinate flipped

    # Plot the connections
    for connection in connections:
        joint1, joint2 = connection
        x1, y1, z1 = coordinates_dict[joint1][0], coordinates_dict[joint1][1], coordinates_dict[joint1][2]
        x2, y2, z2 = coordinates_dict[joint2][0], coordinates_dict[joint2][1], coordinates_dict[joint2][2]
        ax.plot([x1, x2], [-y1, -y2], 'b-')  # 'b-' represents blue lines for connections, y-coordinate flipped

    total_sum = 0
    for i in range(len(old_coordinates) - 1):
        x1, y1, z1 = old_coordinates[i]
        x2, y2, z2 = old_coordinates[i + 1]
        ax.plot([x1, x2], [-y1, -y2],
                color='#4CAC58',  # Set the line color to #4CAC58 (a shade of green)
                linewidth=4,  # Increase the line width to 2
                solid_capstyle='butt')  # Set the line cap style to 'butt' for a straight appearance
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip([x1, x1, z1], [x2, y2, z2])))
        total_sum += (distance/100) # we divide by 100 to get the distance in meters and not cm
    speed = total_sum / (len(old_coordinates) / 30)

    total_sum = round(total_sum, 3)
    path_covered_text = f'Distance: {total_sum} m'
    time_text = f'Time: {round((len(old_coordinates) / 30), 3)} sec'
    speed_text = f'Speed: {round(speed, 3)} m/sec'
    plt.text(-0.9, 0.9, path_covered_text, fontsize=12, color='black')
    plt.text(-0.9, 0.8, time_text, fontsize=12, color='black')
    plt.text(-0.9, 0.7, speed_text, fontsize=12, color='black')

    # Set the axis limits
    ax.set_xlim([-1, 1])  # Adjust the limits as per your requirements
    ax.set_ylim([-1, 1])  # Adjust the limits as per your requirements

    plt.margins(0, 0)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.05, top=0.95)

    return plt


def draw_skeleton_variance(coordinates_dict, connections, mean_position_of_joint, old_coordinates):
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the joints
    for joint in coordinates_dict:
        x = coordinates_dict[joint][0]
        y = coordinates_dict[joint][1]
        z = coordinates_dict[joint][2]

        ax.plot(x, -y, 'ro')  # 'ro' represents red circles for joints, y-coordinate flipped

    # Plot the connections
    for connection in connections:
        joint1, joint2 = connection
        x1, y1, z1 = coordinates_dict[joint1][0], coordinates_dict[joint1][1], coordinates_dict[joint1][2]
        x2, y2, z2 = coordinates_dict[joint2][0], coordinates_dict[joint2][1], coordinates_dict[joint2][2]
        ax.plot([x1, x2], [-y1, -y2], 'b-')  # 'b-' represents blue lines for connections, y-coordinate flipped

        # Plot green lines from all points in old_coordinates to the mean_position_of_joint point
        hand_right_joint_x, hand_right_joint_y, _ = mean_position_of_joint
        for i in range(len(old_coordinates)):
            x_old_coordinates, y_old_coordinates, z_old_coordinates = old_coordinates[i]
            ax.plot([x_old_coordinates, hand_right_joint_x], [-y_old_coordinates, -hand_right_joint_y],
                    color='#4CAC58',  # Set the line color to #4CAC58 (a shade of green)
                    linewidth=3,  # Increase the line width to 2
                    solid_capstyle='butt')  # Set the line cap style to 'butt' for a straight appearance

    # Add a circle patch at the mean_position_of_joint
    circle_radius = 0.03  # Adjust the radius of the circle as needed
    circle = patches.Circle((hand_right_joint_x, -hand_right_joint_y), circle_radius, edgecolor='black',
                            facecolor='black')
    ax.add_patch(circle)

    if len(old_coordinates) > 1:
        coordinates_arr = np.array(old_coordinates)
        cov_matrix = np.cov(coordinates_arr)
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        largest_eigenvalue = np.max(np.real(eigenvalues))

        volume_text = f'Variance: {round(largest_eigenvalue, 3)} cm^2'
        plt.text(-0.9, 0.9, volume_text, fontsize=12, color='black')

    # Set the axis limits
    ax.set_xlim([-1, 1])  # Adjust the limits as per your requirements
    ax.set_ylim([-1, 1])  # Adjust the limits as per your requirements

    plt.margins(0, 0)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.05, top=0.95)

    return plt


def draw_skeleton_volume(coordinates_dict, connections, old_coordinates, hull, volume):
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the joints
    for joint in coordinates_dict:
        x = coordinates_dict[joint][0]
        y = coordinates_dict[joint][1]
        z = coordinates_dict[joint][2]

        ax.plot(x, -y, 'ro')  # 'ro' represents red circles for joints, y-coordinate flipped

    # Plot the connections
    for connection in connections:
        joint1, joint2 = connection
        x1, y1, z1 = coordinates_dict[joint1][0], coordinates_dict[joint1][1], coordinates_dict[joint1][2]
        x2, y2, z2 = coordinates_dict[joint2][0], coordinates_dict[joint2][1], coordinates_dict[joint2][2]
        ax.plot([x1, x2], [-y1, -y2], 'b-')  # 'b-' represents blue lines for connections, y-coordinate flipped

        if len(old_coordinates) > 4:
            # Plotting the convex hull
            for simplex in hull.simplices:
                x_coords = [old_coordinates[i][0] for i in simplex]
                y_coords = [-old_coordinates[i][1] for i in simplex]
                plt.plot(x_coords, y_coords, 'k-')
        else:
            for old_coordinate in old_coordinates:
                x_old, y_old = old_coordinate[0], old_coordinate[1]
                ax.plot(x_old, -y_old, 'ko')  # 'ko' represents black dots, y-coordinate flipped

    # Set the axis limits
    ax.set_xlim([-1, 1])  # Adjust the limits as per your requirements
    ax.set_ylim([-1, 1])  # Adjust the limits as per your requirements

    if volume is not None:
        volume = round(volume, 6)
        volume_text = f'Volume: {volume} cm^3'
        plt.text(-0.9, 0.9, volume_text, fontsize=12, color='black')
    else:
        volume_text = f'Volume: {0.00} cm^3'
        plt.text(-0.9, 0.9, volume_text, fontsize=12, color='black')

    plt.margins(0, 0)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.05, top=0.95)

    return plt


def draw_skeleton_distance_from_body(coordinates_dict, connections, distance, old_distances_queue, hand, NormalVector,
                                     PointOnPlane):
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the joints
    for joint in coordinates_dict:
        x = coordinates_dict[joint][0]
        y = coordinates_dict[joint][1]
        ax.plot(x, -y, 'ro')  # 'ro' represents red circles for joints, y-coordinate flipped

    # Plot the connections
    for connection in connections:
        joint1, joint2 = connection
        x1, y1 = coordinates_dict[joint1][0], coordinates_dict[joint1][1]
        x2, y2 = coordinates_dict[joint2][0], coordinates_dict[joint2][1]
        ax.plot([x1, x2], [-y1, -y2], 'b-')  # 'b-' represents blue lines for connections, y-coordinate flipped

    # Draw the plane
    if NormalVector is not None and PointOnPlane is not None:
        # Define the plane equation ax + by + cz + d = 0
        a, b, c = NormalVector
        d = -PointOnPlane.dot(NormalVector)

        # Define a rectangle that represents the plane
        plane_width = 2  # Adjust the plane width as needed
        plane_height = 2  # Adjust the plane height as needed
        plane_points = np.array([
            [-plane_width / 2, -plane_height / 2],
            [plane_width / 2, -plane_height / 2],
            [plane_width / 2, plane_height / 2],
            [-plane_width / 2, plane_height / 2]
        ])

        # Rotate and translate the plane to fit the body orientation
        R = np.array([[a, b], [-b, a]])
        plane_points_rotated = np.dot(plane_points, R.T)
        plane_points_translated = plane_points_rotated + [PointOnPlane[0], -PointOnPlane[1]]

        # Create and plot the plane polygon
        plane_polygon = Polygon(plane_points_translated, edgecolor='g', facecolor='green', alpha=0.5)
        ax.add_patch(plane_polygon)

        # Plot the distance line from 'hand' to its projection on the plane
        projection = hand - (hand.dot(NormalVector) + d) * NormalVector  # Projection on the plane
        ax.plot([hand[0], projection[0]], [-hand[1], -projection[1]], color='black', linestyle='-',
                linewidth=4)  # 'k--' represents black dashed line


    # Set the axis limits
    ax.set_xlim([-1, 1])  # Adjust the limits as per your requirements
    ax.set_ylim([-1, 1])  # Adjust the limits as per your requirements

    distance = round(distance, 3)
    mean_distance = round(mean(old_distances_queue),3)
    variance1 = round(np.var(old_distances_queue),3)

    distance_text = f'Distance: {distance} cm'
    mean_text = f'Mean Distance: {mean_distance} cm'
    variance1_text = f'Variance: {variance1} cm^2'

    plt.text(-0.9, 0.9, distance_text, fontsize=12, color='black')
    plt.text(-0.9, 0.8, mean_text, fontsize=12, color='black')
    plt.text(-0.9, 0.7, variance1_text, fontsize=12, color='black')


    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.05, top=0.95)

    return plt


def draw_image_next_to_skeleton(image_path):
    # Open and load the images
    skeleton_image = plt.imread("skeleton.jpg")
    image = Image.open(image_path)

    # Create a new figure and subplot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display the images on the subplots
    axs[0].imshow(skeleton_image)
    axs[0].axis('off')  # Turn off axis labels

    axs[1].imshow(image)
    axs[1].axis('off')  # Turn off axis labels

    # Minimize margins and spacing while keeping space for axes
    plt.margins(0, 0)
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0)

    return plt


def calculate_distance_hand_from_body_one_frame(Body1, Body2, Body3, hand):
    # Body coordinates
    Body1 = np.array(Body1)
    Body2 = np.array(Body2)
    Body3 = np.array(Body3)

    # Hand coordinates
    Hand = np.array(hand)  # Replace x_hand, y_hand, z_hand with the coordinates of the hand

    # Create vectors
    Vector1 = Body2 - Body1
    Vector2 = Body3 - Body1

    # Calculate normal vector
    NormalVector = np.cross(Vector1, Vector2)
    NormalVector = NormalVector / np.linalg.norm(NormalVector)  # Normalize the normal vector

    # Calculate the distance from the point to the plane
    distance = np.abs(np.dot(NormalVector, Hand - Body1)) / np.linalg.norm(NormalVector)

    # Define a point on the plane (Body1)
    PointOnPlane = Body1

    return distance, NormalVector, PointOnPlane