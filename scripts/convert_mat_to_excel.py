import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
import os
import shutil
import scipy.io

# read mat file
# matname = r'D:\Howie_FM2_Brain_Data\participant_trait_and_experiment_data.mat'
# matname = r'Y:\pain_data_sets_brain\Howie_FM2_Brain_Data\participant_trait_and_experiment_data.mat'
matname = r'F:\pain_data_sets\TwoPain\TwoPain_database.mat'

sheetname = 'datarecord'
# fieldnames = {'participantid', 'sex', 'gender', 'handedness', 'age', 'bmi', 'studygroup', 'STAI_Y_1', 'STAI_Y_2', 'SDS', 'BDS', 'PCS_Total', 'PCS_Rum', \
# 'PCS_Mag', 'PCS_Help', 'FIQR_SIQR', 'COMPASS', 'SFMPQ_Total', 'SFMPQ_Cont', 'SFMPQ_Inter', 'SFMPQ_Neur', 'SFMPQ_Aff', 'criteria_total', \
# 'wpi', 'ss', 'three_months', 'other_cond', 'average_temp', 'average_pain', 'pain_sensitivity', 'exclude'}

p,f1 = os.path.split(matname)
f,e = os.path.splitext(f1)
output_name = os.path.join(p,f+'.xlsx')

matdata = scipy.io.loadmat(matname, squeeze_me=True)
data = matdata[sheetname]
pdata=pd.DataFrame(data)

with pd.ExcelWriter(output_name, engine="openpyxl", mode='w') as writer:
	pdata.to_excel(writer, sheet_name=sheetname)
