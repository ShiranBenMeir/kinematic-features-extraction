import os
import pandas as pd
import Utilitis

"""
    open all Body folders inside the split sessions and store the "body" txt files (one file for each frame)
    in the dictionary "body_dict". The keys in the dictionary are the word/sign and the values are the 
    all paths of all the txt files in that word/sign folder.

    @param directory: the path to split sessions directory.
    @type directory: string

    @return: body_dict. 
"""


def open_body_folders_from_split_sessions(directory):
    body_dict = {}
    dominant_hand_dict = {}
    # the loop travels through all files in all Body folders inside split sessions directory.
    for root, dirs, files in os.walk(directory):
        if 'Body' in dirs:
            # extract the name/sign (the name of the folder inside split sessions directory)
            word_name = Utilitis.extract_word_name_from_directory_path(root)
            dominant_hand = Utilitis.extract_dominant_hand_name_from_directory_path(root)
            body_folder = os.path.join(root, 'Body')
            # go through each txt file in Body folder
            for body_root, body_dirs, body_files_inner in os.walk(body_folder):
                # set empty list that will store all the txt file in a specific word/sign folder
                body_files = []
                for file in body_files_inner:
                    file_path = os.path.join(body_root, file)
                    body_files.append(file_path)
            body_dict[word_name] = body_files
            dominant_hand_dict[word_name] = dominant_hand
    return body_dict, dominant_hand_dict


"""
    open all Color folders inside the split sessions and store the "Color" JPEG files (one file for each frame)
    in the dictionary "color_dict". The keys in the dictionary are the word/sign and the values are the 
    all paths of all the JPEG files in that word/sign folder.

    @param directory: the path to split sessions directory.
    @type directory: string

    @return: color_dict. 
"""


def open_color_folders_from_split_sessions(directory):
    color_dict = {}
    # the loop travels through all files in all Color folders inside split sessions directory.
    for root, dirs, files in os.walk(directory):
        if 'Color' in dirs:
            # extract the name/sign (the name of the folder inside split sessions directory)
            word_name = Utilitis.extract_word_name_from_directory_path(root)
            color_folder = os.path.join(root, 'Color')
            # go through each JPEG file in Color folder
            for color_root, color_dirs, color_files_inner in os.walk(color_folder):
                # set empty list that will store all the JPEG file in a specific word/sign folder
                color_files = []
                for file in color_files_inner:
                    file_path = os.path.join(color_root, file)
                    color_files.append(file_path)
            color_dict[word_name] = color_files
    return color_dict


"""
    The function reads one txt body file (one frame) and extracts info from this file.
    The info is: the name of the joints, the coordinates of all joints in the frame, 
    the frame_confidence which is the confidence vale (between 0 to 2)  of the coordinate
    for each joint in the frame
    each joint and its coordinate and confidence is stored as a tuple
     in a list named list_of_tuples_pre_row_in_frame_file.


    @param file_path: the path of a single txt file in a Body folder.
    @type directory: string

    @return: list_of_tuples_pre_row_in_frame_file - A list that contains tuples. each tuple represent the information
              about one joint in the frame. fro example: ('Pelvis', [0.103234146118164, -0.0618840713500977, 1.97004956054688], '2') 

"""


def open_one_body_file(file_path):
    list_of_tuples_pre_row_in_frame_file = []
    # Open the text file for reading
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Split the line using '#' as the delimiter
            parts = line.strip().split('#')
            # Extract the coordinate values, name, and frame confidence
            coordinates = [float(part) for part in parts[:3]]
            name = parts[4]
            frame_confidence = parts[3]

            list_of_tuples_pre_row_in_frame_file.append((name, coordinates, frame_confidence))

    return list_of_tuples_pre_row_in_frame_file


"""
   The function gets a path of one txt file (a frame) and extract info.
   the extracted info is the name of the file (name of the frame) and the  ID (for example: B16_CC_01)

    @param file_path: the path of a frame (txt file in Body folder).
    @type directory: string

    @return fil_id: for example- B16_CC_01
    @return file name: name of the frame 
"""


def extract_file_info(file_path):
    file_name = os.path.basename(file_path).split('.')[0]
    parent_folder_path = os.path.dirname(file_path)
    name_of_folder_inside_split_sessions = os.path.basename(os.path.dirname(parent_folder_path))
    split_parts = name_of_folder_inside_split_sessions.split('_')
    split_parts_connected = list(filter(None, split_parts))
    file_id = '_'.join(split_parts_connected[:3])
    return file_id, file_name


"""
   The function writes the name of columns in an xlsx file. 
   The function gets a sheet to write to and a list of tuples. each tuple contains 
   the name of the joint, the coordinate of the joint and the confidence of the coordinate (all for a 
   specific frame). We use the tuples to name the columns with the joint names in adittion to the names 
   we set in "column_names".

    @param list_of_tuples_pre_row_in_frame_file: list of tuples
    @type directory: string

    @param sheet: the xlsx to write the column names to.
    @type: workbook object 

    @return sheet: the xlsx with the column names
"""


def add_col_names_to_xlsx(list_of_tuples_pre_row_in_frame_file, sheet):
    column_names = ["ID", "Word", "File Name", "Dominant Hand"]

    for name, _, _ in list_of_tuples_pre_row_in_frame_file:
        column_names.append(name)
        column_names.append(name + " confidence")

    # Write column names to the first row
    sheet.insert_rows(1)
    for col_num, column_name in enumerate(column_names, start=1):
        sheet.cell(row=1, column=col_num, value=column_name)
    return sheet


"""
   The function create a df with column names for the data_per_video_excel

    @return df
"""


def set_data_per_video_excel():
    DF = pd.DataFrame(columns=[
        "ID",
        'Word',
        "Dominant Hand",
        "Number of frames",
        "Eye average position",
        "Eye variance",
        "Eye volume",
        "Eye speed",
        "Eye sum of distance",
        "Average distance- head (middle of the eyes) from body",
        "Spine chest average position",
        "Spine chest variance",
        "Spine chest volume",
        "Spine chest speed",
        "Spine chest sum of distance",
        "Average distance- spine chest from body",
        "Dominant hand average position",
        "Dominant hand variance",
        "Dominant hand volume",
        "Dominant hand speed",
        "Dominant hand sum of distance",
        "Average distance- dominant hand from body"])

    return DF

