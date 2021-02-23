import pandas as pd
import random

PATH_to_repo =  "C:/Users/nilso/Documents/EPFL/MA4/PDS Turberg/"
PATH_data = PATH_to_repo + 'Rock_Cuttings_Characterisation/Data/'
PATH_save = PATH_to_repo + 'Rock_Cuttings_Characterisation/PostProcessing/Model_Comparison/'

df_test_csv = pd.read_csv(PATH_data+'test/csv_resized_128/test_resized_128_final.csv',index_col=0)
list_im_test = []
random.seed(0)
for i in range(5):
    list_im_test+=random.sample(list(df_test_csv[df_test_csv['rock_type'] ==i].index.values),k = 40)
    
df_test_csv.loc[list_im_test].sort_index().reset_index(drop=True)[['path','rock_type']].to_csv(PATH_save + 'comparison_data.csv')