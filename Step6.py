import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import FeaturesCalculation
import FilesUilitis
import Utilitis
import cv2
import glob


'''''
TRACKED_JOINT = 'HandRight'
SPLIT_SESSIONS_DIRECTORY_PATH = "splitted_sessions"
OUTPUT_DIRECTORY_PATH = "C:/Users/shiran/multiplefiles_Step5_Step6/output_skeletons"
DATA_PER_FRAME_EXCEL_PATH = "data_per_frame.xlsx"
DATA_PER_VIDEO_EXCEL_PATH = "data_per_video.xlsx"
'''
VIDEO_TYPE = "variance"

"""
    create_skeletons - create JPEG images. each image represent one frame. the images consisted of the frame image next
    to a skeleton of that frame image.

"""
def create_skeletons(FILE_NAME, SPLIT_SESSIONS_DIRECTORY_PATH, DATA_PER_FRAME_EXCEL_PATH, DATA_PER_VIDEO_EXCEL_PATH, OUTPUT_DIRECTORY_PATH, TRACKED_JOINT):
    data_per_frame_df = pd.read_excel(DATA_PER_FRAME_EXCEL_PATH)
    data_per_frame_df = Utilitis.convert_coordinates_to_float(data_per_frame_df)
    data_per_video_df = pd.read_excel(DATA_PER_VIDEO_EXCEL_PATH)

    # set the name of the joints we want to see on the skeleton
    joints = Utilitis.set_joints()
    # set the connections between joint (the connection lines on skeleton between the chosen joints
    connections = Utilitis.set_connections()
    # creating image_dict -  the values are the words/signs and the values are lists of paths to color images.
    image_dict = FilesUilitis.open_color_folders_from_split_sessions(SPLIT_SESSIONS_DIRECTORY_PATH)



    word_counter = 1
    # iterate over all image_dict values - the lists of color images paths
    for word, image_paths_list in image_dict.items():
        print("word/sign: "+ str(word_counter) + " out of " + str(len(image_dict.items())))
        word_counter = word_counter + 1
        # open new folders named by the word/signs in OUTPUT_DIRECTORY_PATH
        word_folder_path = os.path.join(OUTPUT_DIRECTORY_PATH, FILE_NAME, word)
        os.makedirs(word_folder_path)

        old_distances_queue_hand_from_body = []
        old_distances_queue_hand_from_body_average_plane = []
        simulation_types = ["path covered", "variance", "volume", "distance hand from body", "distance hand from body average plane"]
        for simulation_type in simulation_types:
            old_coordinates = []  # list of previous coordinates of TRACKED_JOINT

            frame_counter = 1
            for image_path in image_paths_list:
                print("Generating " + simulation_type + " " + str(frame_counter) + " out of " + str(len(image_paths_list)))
                frame_counter = frame_counter + 1
                frame_name = os.path.basename(image_path).split('.')[0]
                # coordinates_dict - name of joint as key, coordinates as values
                coordinates_dict = Utilitis.extract_skeleton_coordinates(data_per_frame_df, frame_name, joints)
                # save previous coordinates to generate track
                old_coordinates.append(coordinates_dict[TRACKED_JOINT])
                if simulation_type == "path covered":
                    skeleton_plt = Utilitis.draw_skeleton(coordinates_dict, connections, old_coordinates)

                if simulation_type == "variance":
                    mean_position_of_joint = \
                        data_per_video_df.loc[data_per_video_df["Word"] == word, "Dominant hand average position"].iloc[0]
                    mean_position_of_joint = Utilitis.convert_one_coordinate_to_float(mean_position_of_joint)
                    skeleton_plt = Utilitis.draw_skeleton_variance(coordinates_dict, connections, mean_position_of_joint,
                                                                   old_coordinates)
                if simulation_type == "volume":
                    if len(old_coordinates) > 4:
                        volume, hull = FeaturesCalculation.volume_calculation(old_coordinates)
                    else:
                        hull = None
                        volume = None
                    skeleton_plt = Utilitis.draw_skeleton_volume(coordinates_dict, connections, old_coordinates, hull,
                                                                 volume)

                if simulation_type == "distance hand from body":
                    # Body coordinates
                    Body1 = np.array(coordinates_dict["SpineNavel"])
                    Body2 = np.array(coordinates_dict["ClavicleRight"])
                    Body3 = np.array(coordinates_dict["ClavicleLeft"])
                    hand = np.array(coordinates_dict[TRACKED_JOINT])
                    distance, NormalVector, PointOnPlane = Utilitis.calculate_distance_hand_from_body_one_frame(Body1,
                                                                                                                Body2,
                                                                                                                Body3, hand)
                    old_distances_queue_hand_from_body.append(distance)
                    skeleton_plt = Utilitis.draw_skeleton_distance_from_body(coordinates_dict, connections,distance, old_distances_queue_hand_from_body,
                                                                             hand, NormalVector, PointOnPlane)
                if simulation_type == "distance hand from body average plane":
                    word_df = data_per_frame_df[data_per_frame_df['Word'] == word].reset_index(drop=True)
                    Body1, Body2, Body3 = Utilitis.average_body_coordinates_calculation(word_df)
                    hand = np.array(coordinates_dict[TRACKED_JOINT])
                    distance, NormalVector, PointOnPlane = Utilitis.calculate_distance_hand_from_body_one_frame(Body1,
                                                                                                                Body2,
                                                                                                                Body3, hand)
                    old_distances_queue_hand_from_body.append(distance)

                    skeleton_plt = Utilitis.draw_skeleton_distance_from_body(coordinates_dict, connections,distance, old_distances_queue_hand_from_body,
                                                                             hand, NormalVector, PointOnPlane)

                skeleton_path = os.path.join(OUTPUT_DIRECTORY_PATH)
                skeleton_plt.savefig('skeleton.jpg', dpi=300, format='jpg')
                skeleton_plt.close()
                img_skeleton_plt = Utilitis.draw_image_next_to_skeleton(image_path)
                os.remove('skeleton.jpg')
                skeleton_plt.savefig(os.path.join(word_folder_path, f"{frame_name}.jpg"))

                skeleton_plt.savefig('skeleton.jpg', dpi=300, format='jpg')
                # plt.show()
                skeleton_plt.close('all')
                plt.close()


            simulation_type_path = os.path.join(OUTPUT_DIRECTORY_PATH, FILE_NAME, simulation_type)
            images_to_video(word_folder_path, word, FILE_NAME, simulation_type, simulation_type_path, fps=7)
            skeleton_plt.close('all')




def images_to_video(word_folder_path, word, FILE_NAME, simulation_type, simulation_type_path, fps=7):
    image_files = [f for f in os.listdir(word_folder_path) if f.endswith('.jpg')]
    image_files.sort()  # Sort the image files to maintain order

    # Initialize the video writer
    video_path = os.path.join(word_folder_path, f"{simulation_type + ' ' + word + '_' + FILE_NAME}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (1000, 500))  # Use the desired frame size

    for skeleton_img_file in image_files:
        skeleton_img_file_path = os.path.join(word_folder_path, skeleton_img_file)

        # Read the image using the appropriate library (OpenCV in this case)
        img = cv2.imread(skeleton_img_file_path)

        # Resize the image to match the video frame size
        img = cv2.resize(img, (1000, 500))

        video_writer.write(img)

    # Release the video writer
    video_writer.release()

    jpeg_files = glob.glob(os.path.join(word_folder_path, "*.jpg"))
    for jpeg_file in jpeg_files:
        os.remove(jpeg_file)




def main(FILE_NAME, SPLIT_SESSIONS_DIRECTORY_PATH, DATA_PER_FRAME_EXCEL_PATH, DATA_PER_VIDEO_EXCEL_PATH,
                   OUTPUT_DIRECTORY_PATH, TRACKED_JOINT):
    print("main6")
    create_skeletons(FILE_NAME, SPLIT_SESSIONS_DIRECTORY_PATH, DATA_PER_FRAME_EXCEL_PATH, DATA_PER_VIDEO_EXCEL_PATH, OUTPUT_DIRECTORY_PATH, TRACKED_JOINT)
    #images_to_video(OUTPUT_DIRECTORY_PATH)


if __name__ == '__main__':
    main()