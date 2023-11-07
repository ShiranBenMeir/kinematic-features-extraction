import Step5
import Step6
import pandas as pd


def execute_Step5():
    df_parameters_for_step5 = pd.read_excel('parameters!.xlsx', 'Step5')
    for index, row in df_parameters_for_step5.iterrows():
        FILE_NAME = row[0]
        SPLIT_SESSIONS_DIRECTORY_PATH = row[1]
        OUTPUT_PATH = row[2]

        print("Analysis of: " + FILE_NAME)
        Step5.main(SPLIT_SESSIONS_DIRECTORY_PATH, OUTPUT_PATH)



def execute_Step6():
    df_parameters_for_step5 = pd.read_excel('parameters.xlsx', 'Step6')
    for index, row in df_parameters_for_step5.iterrows():
        FILE_NAME = row[0]
        SPLIT_SESSIONS_DIRECTORY_PATH = row[1]
        DATA_PER_FRAME_EXCEL_PATH = row[2]
        DATA_PER_VIDEO_EXCEL_PATH = row[3]
        OUTPUT_DIRECTORY_PATH = row[4]
        TRACKED_JOINT = row[5]

        print("Generating simulation for: "+ FILE_NAME)
        Step6.main(FILE_NAME,SPLIT_SESSIONS_DIRECTORY_PATH, DATA_PER_FRAME_EXCEL_PATH, DATA_PER_VIDEO_EXCEL_PATH,
                   OUTPUT_DIRECTORY_PATH, TRACKED_JOINT)


if __name__ == '__main__':
    print("main")
    execute_Step5()
    execute_Step6()


