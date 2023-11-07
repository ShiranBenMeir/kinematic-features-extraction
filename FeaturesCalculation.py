import math
import numpy as np
from scipy.spatial import ConvexHull
import Utilitis


def average_position_calculation(coordinates_lst):
    # Calculate the sum of coordinates
    total_sum = [sum(coord) for coord in zip(*coordinates_lst)]

    # Calculate the average
    average_coordinate = [x / len(coordinates_lst) for x in total_sum]

    return average_coordinate


def variance_position_calculation(coordinates_lst):
    coordinates_arr = np.array(coordinates_lst)
    cov_matrix = np.cov(coordinates_arr)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    largest_eigenvalue = np.max(np.real(eigenvalues))
    return largest_eigenvalue

def volume_calculation(coordinates_lst):
    points = np.array(coordinates_lst, dtype=np.float64)  # Convert column to a NumPy array
    hull = ConvexHull(points)
    volume = hull.volume
    return volume, hull

def another_volume_calculation(df):
    coordinates_ndarray = df['SpineChest'].values

    # Extract x, y, and z values into separate lists
    x_values = []
    y_values = []
    z_values = []

    for coord in coordinates_ndarray:
        x_values.append(coord[0])
        y_values.append(coord[1])
        z_values.append(coord[2])

    y = "hi"
    points = np.array(list(zip(x_values, y_values, z_values)))
    volume = ConvexHull(points).volume

    return volume


def sum_of_distance(coordinates_lst):
    # Initialize variables
    total_sum = 0

    # Calculate the sum of Euclidean distances
    for i in range(len(coordinates_lst) - 1):
        current_coord = coordinates_lst[i]
        next_coord = coordinates_lst[i + 1]
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_coord, next_coord)))
        total_sum += distance

    return total_sum



def speed_calculation(coordinates_lst, distance_sum):
    first_coor = coordinates_lst[0]
    last_coor = coordinates_lst[-1]

    sum_of_squares_first_and_last = sum((c2 - c1) ** 2 for c1, c2 in zip(first_coor, last_coor))
    euclidean_distance_first_and_last = math.sqrt(sum_of_squares_first_and_last)
    time_of_video_in_sec = len(coordinates_lst)/30
    average_speed = distance_sum / time_of_video_in_sec

    return average_speed


def distance_hand_from_body_calculation(df, DOMINANT_HAND):
    distance_sum = 0
    for index, row in df.iterrows():
        #mean_SpineNavel, mean_ClavicleRight, mean_ClavicleLeft =Utilitis.average_plane_coordinates_calculation(df)

        # Body coordinates
        Body1 = np.array(row['SpineNavel'])
        Body2 = np.array(row['ClavicleRight'])
        Body3 = np.array(row['ClavicleLeft'])

        #average_hand_coor = average_position_calculation(df.loc[0,DOMINANT_HAND])
        # Hand coordinates
        Hand_col_name = Utilitis.donminant_hand_filter(DOMINANT_HAND)
        Hand = np.array(row[Hand_col_name])  # Replace x_hand, y_hand, z_hand with the coordinates of the hand

        # Create vectors
        Vector1 = Body2 - Body1
        Vector2 = Body3 - Body1

        # Calculate normal vector
        NormalVector = np.cross(Vector1, Vector2)
        NormalVector = NormalVector / len(NormalVector)

        # Calculate the distance from the point to the plane
        distance = np.abs(np.dot(NormalVector, Hand - Body1)) / np.linalg.norm(NormalVector)
        distance_sum = distance_sum + distance

        # Define a point (for step 6)
        PointOnPlane = Body1

    average_distance = distance_sum / len(df)
    return average_distance

def distance_SpineChest_from_body_calculation(df):
    Body1, Body2, Body3= Utilitis.average_body_coordinates_calculation(df)
    distance_sum = 0
    for index, row in df.iterrows():
        # Hand coordinates
        SpineChest = np.array(row['SpineChest'])

        # Create vectors
        Vector1 = Body2 - Body1
        Vector2 = Body3 - Body1

        # Calculate normal vector
        NormalVector = np.cross(Vector1, Vector2)
        NormalVector = NormalVector / len(NormalVector)

        # Calculate the distance from the point to the plane
        distance = np.abs(np.dot(NormalVector, SpineChest - Body1)) / np.linalg.norm(NormalVector)
        distance_sum = distance_sum + distance

    average_distance = distance_sum / len(df)
    return average_distance


def distance_head_from_body_calculation(df, middle_of_the_eyes_column):
    distance_sum = 0
    for index, row in df.iterrows():

        # Body coordinates
        Body1 = np.array(row['SpineNavel'])
        Body2 = np.array(row['ClavicleRight'])
        Body3 = np.array(row['ClavicleLeft'])

        Head = np.array(middle_of_the_eyes_column[index])

        # Create vectors
        Vector1 = Body2 - Body1
        Vector2 = Body3 - Body1

        # Calculate normal vector
        NormalVector = np.cross(Vector1, Vector2)
        NormalVector = NormalVector / len( NormalVector)

        # Calculate the distance from the point to the plane
        distance = np.abs(np.dot(NormalVector, Head - Body1)) / np.linalg.norm(NormalVector)
        distance_sum = distance_sum + distance

    average_distance = distance_sum / len(df)
    return average_distance