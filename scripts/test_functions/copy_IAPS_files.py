import pandas as pd
import os
import shutil

sourcedir = r'Y:\IAPS_image_database\IAPS 1-20 Images'

# neutral
xlname = r'Y:\IAPS_image_database\ALL_Neutral_Visual_Stimuli_n260.xlsx'
destdir = r'Y:\IAPS_image_database\Neutral'

# negative
xlname = r'Y:\IAPS_image_database\ALL_Negative_Visual_Stimuli_n260.xlsx'
destdir = r'Y:\IAPS_image_database\Negative'

# positive
xlname = r'Y:\IAPS_image_database\ALL_Positive_Visual_Stimuli_n260.xlsx'
destdir = r'Y:\IAPS_image_database\Positive'

xls = pd.ExcelFile(xlname, engine='openpyxl')
df1 = pd.read_excel(xls, 'Sheet1', header = 2)
iaps_names = df1.loc[:, 'IAPS']

for name in iaps_names:
	source_name = os.path.join(sourcedir, str(name) + '.jpg')
	dest_name = os.path.join(destdir, str(name) + '.jpg')
	try:
		shutil.copyfile(source_name, dest_name)
		print('copied from {} to {}'.format(source_name, dest_name))
	except:
		print('.....................could not copy from {} to {}'.format(source_name, dest_name))

