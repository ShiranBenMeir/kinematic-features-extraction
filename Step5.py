import openpyxl
import pandas as pd
import FeaturesCalculation
import FilesUilitis
import Utilitis

#SPLIT_SESSIONS_DIRECTORY_PATH = "splitted_sessions"
#DOMINANT_HAND = 'Right'

"""
    create_data_per_frame_excel - create an excel file with all the given info about each frame in each word/sign video.
    first column: ID -  ID number of the file ( for example B16_CC_01).
    second column: Word - The word/sign that the person was signing
    third column: File Name - te file name of the frame (in Body folder)
    forth column: Dominant Hand - the dominant hand of the signer
    fifth column to the last column: joints - the coordinate of each joint for a specific frame. 

    """
def create_data_per_frame_excel(SPLIT_SESSIONS_DIRECTORY_PATH, OUTPUT_PATH):
    # set the xslx file we will write to
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    # Specify the directory we want to traverse
    body_dict, dominant_hand_dict = FilesUilitis.open_body_folders_from_split_sessions(SPLIT_SESSIONS_DIRECTORY_PATH)

    # loop through each word/sign folder
    for word in body_dict:
        dominantHand = dominant_hand_dict[word]
        all_body_files_of_specific_word = body_dict[word]

        # loop through each body file in the word/sign folder
        for body_file in all_body_files_of_specific_word:
            # create a list of tuples. each tuple contains: (name, coordinates, frame_confidence) for each file (frame)
            list_of_tuples_per_row_in_frame_file = FilesUilitis.open_one_body_file(body_file)

            file_id, file_name = FilesUilitis.extract_file_info(body_file)
            # Create the row data- the row that will be written to the xlsx
            row_data = [file_id, word, file_name, dominantHand]
            # Append coordinates and confidence to row_data
            for name, coordinates, confidence in list_of_tuples_per_row_in_frame_file:
                coordinates_str = ','.join(str(float(c)) for c in coordinates)
                row_data.append(coordinates_str)
                row_data.append(int(confidence))

            # Add the row to the sheet
            sheet.append(row_data)

    FilesUilitis.add_col_names_to_xlsx(list_of_tuples_per_row_in_frame_file, sheet)

    data_per_frame_file_name = OUTPUT_PATH+ '/' + file_id + ' data_per_frame.xlsx'
    # Save the workbook
    workbook.save(data_per_frame_file_name)
    # Close the workbook
    workbook.close()
    return data_per_frame_file_name, dominant_hand_dict


def create_features_dataframe(df, DOMINANT_HAND):
    # Convert numbers inside vectors to float
    df = Utilitis.convert_coordinates_to_float(df)

    ######################
    ####### eyes #########
    ######################
    middle_of_the_eyes_column = Utilitis.middle_of_the_eyes_calculation(df, ['EyeLeft', 'EyeRight'])
    eye_average_position = FeaturesCalculation.average_position_calculation(middle_of_the_eyes_column)
    eye_std_of_positions = FeaturesCalculation.variance_position_calculation(middle_of_the_eyes_column)
    eye_volume,_ = FeaturesCalculation.volume_calculation(middle_of_the_eyes_column)
    eye_distance_sum = FeaturesCalculation.sum_of_distance(middle_of_the_eyes_column)
    eye_speed = FeaturesCalculation.speed_calculation(middle_of_the_eyes_column, eye_distance_sum)
    mean_head_from_body_dist = FeaturesCalculation.distance_head_from_body_calculation(df, middle_of_the_eyes_column)


    ######################
    ##### spine chest ####
    ######################
    spineChest_column = list(df['SpineChest'])
    spineChest_average_position = FeaturesCalculation.average_position_calculation(spineChest_column)
    spineChest_std_of_positions = FeaturesCalculation.variance_position_calculation(spineChest_column)
    spineChest_volume,_ = FeaturesCalculation.volume_calculation(spineChest_column)
    another_spineChest_volume = FeaturesCalculation.another_volume_calculation(df)
    spineChest_distance_sum = FeaturesCalculation.sum_of_distance(spineChest_column)
    spineChest_speed = FeaturesCalculation.speed_calculation(spineChest_column, spineChest_distance_sum)
    mean_SpineChest_from_body_dist = FeaturesCalculation.distance_SpineChest_from_body_calculation(df)



    ######################
    #### dominant hand ###
    ######################
    dominanatHand_col_name = Utilitis.donminant_hand_filter(DOMINANT_HAND)
    dominantHand_column = list(df[dominanatHand_col_name])
    dominantHand_average_position = FeaturesCalculation.average_position_calculation(dominantHand_column)
    dominantHand_std_of_positions = FeaturesCalculation.variance_position_calculation(dominantHand_column)
    dominantHand_volume,_ = FeaturesCalculation.volume_calculation(dominantHand_column)
    dominantHand_distance_sum = FeaturesCalculation.sum_of_distance(dominantHand_column)
    dominantHand_speed = FeaturesCalculation.speed_calculation(dominantHand_column, dominantHand_distance_sum)
    mean_hand_from_body_dist= FeaturesCalculation.distance_hand_from_body_calculation(df, DOMINANT_HAND)



    features_dict = {
        "ID": str(df['ID'][0]),
        'Word': str(df['Word'][0]),
        "Dominant Hand": str(df['Dominant Hand'][0]),
        "Number of frames": len(df),
        "Eye average position": ",".join(map(str, eye_average_position)),
        "Eye variance": eye_std_of_positions,
        "Eye volume": eye_volume,
        "Eye speed": eye_speed,
        "Eye sum of distance": eye_distance_sum,
        "Average distance- head (middle of the eyes) from body": mean_head_from_body_dist,
        "Spine chest average position": ",".join(map(str, spineChest_average_position)),
        "Spine chest variance": spineChest_std_of_positions,
        "Spine chest volume": spineChest_volume,
        "Spine chest speed": spineChest_speed,
        "Spine chest sum of distance": spineChest_distance_sum,
        "Average distance- spine chest from body": mean_SpineChest_from_body_dist,
        "Dominant hand average position": ",".join(map(str, dominantHand_average_position)),
        "Dominant hand variance": dominantHand_std_of_positions,
        "Dominant hand volume": dominantHand_volume,
        "Dominant hand speed": dominantHand_speed,
        "Dominant hand sum of distance": dominantHand_distance_sum,
        "Average distance- dominant hand from body": mean_hand_from_body_dist,
    }

    # Create a DataFrame from the dictionary
    features_df = pd.DataFrame.from_dict(features_dict, orient='index').transpose()

    return features_df




"""
    create_data_per_video_excel - create an excel file with all the given info about wach word/sign video.
    first column: ID -  ID number of the file ( for example B16_CC_01).
    second column: Word - The word/sign that the person was signing
    third column: Dominant Hand - the dominant hand of the signer
    forth column to the last column: joints - calculated features. 

"""
def create_data_per_video_excel(data_per_frame_file_name, OUTPUT_PATH, dominant_hand_dict):

    data_per_video = FilesUilitis.set_data_per_video_excel()
    data_per_frame = pd.read_excel(data_per_frame_file_name)

    # get all the words from data_per_frame df
    unique_words_from_per_frame = data_per_frame['Word'].unique()
    data_per_frame_grouped_by_word = data_per_frame.groupby('Word')
    for word in unique_words_from_per_frame:
        dominantHand = dominant_hand_dict[word]
        df = data_per_frame_grouped_by_word.get_group(word) # df is the data_per_frame but only with rows of the specific word
        df = df.reset_index(drop=True)  # Resetting index of df
        # calculate the features
        new_df = create_features_dataframe(df, dominantHand)

        # contact the new df to the rest of the df's so each word will be a row in the data_per_video xlsx
        data_per_video = pd.concat([data_per_video, new_df], ignore_index=True)


    file_id = data_per_frame['ID'].unique()[0]
    data_per_video_file_name = OUTPUT_PATH + '/' + file_id + ' data_per_video.xlsx'
    data_per_video.to_excel(data_per_video_file_name, index=False)




def main(SPLIT_SESSIONS_DIRECTORY_PATH, OUTPUT_PATH):
    print("main5")
    data_per_frame_file_name, dominant_hand_dict = create_data_per_frame_excel(SPLIT_SESSIONS_DIRECTORY_PATH,OUTPUT_PATH)
    create_data_per_video_excel(data_per_frame_file_name,OUTPUT_PATH, dominant_hand_dict)


if __name__ == '__main__':
    main()