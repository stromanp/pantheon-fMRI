# sys.path.append(r'C:\Users\Stroman\PycharmProjects\pantheon\venv')
import pysapm
import numpy as np
import copy

data_list = []
results_file_list = [r'E:\SAPMresults_Dec2022\AllPain_1202023213_L1_ref_DBsig.xlsx',
					r'E:\SAPMresults_Dec2022\Pain_1202023213_L1_ref_DBsig.xlsx',
					r'E:\SAPMresults_Dec2022\High_1202023213_L1_ref_DBsig.xlsx',
					r'E:\SAPMresults_Dec2022\RSstim_1202023213_L1_ref_DBsig.xlsx']
cnums = [1, 2, 0, 2, 0, 2, 3, 2, 1, 3]
for nn in range(4):
	data_list.append({'results_file':results_file_list[nn], 'cnums':cnums})


results_file_list = [r'E:\SAPMresults_Dec2022\AllPain_3332340131_L1_ref_DBsig.xlsx',
					r'E:\SAPMresults_Dec2022\Pain_3332340131_L1_ref_DBsig.xlsx',
					r'E:\SAPMresults_Dec2022\High_3332340131_L1_ref_DBsig.xlsx',
					r'E:\SAPMresults_Dec2022\RSstim_3332340131_L1_ref_DBsig.xlsx']
cnums = [3, 3, 3, 2, 3, 4, 0, 1, 3, 1]
for nn in range(4):
	data_list.append({'results_file':results_file_list[nn], 'cnums':cnums})


results_file_list = [r'E:\SAPMresults_Dec2022\AllPain_4203023313_L1_ref_DBsig.xlsx',
					r'E:\SAPMresults_Dec2022\Pain_4203023313_L1_ref_DBsig.xlsx',
					r'E:\SAPMresults_Dec2022\High_4203023313_L1_ref_DBsig.xlsx',
					r'E:\SAPMresults_Dec2022\RSstim_4203023313_L1_ref_DBsig.xlsx']
cnums = [4, 2, 0, 3, 0, 2, 3, 3, 1, 3]

for nn in range(4):
	data_list.append({'results_file':results_file_list[nn], 'cnums':cnums})

for nn in range(len(data_list)):
	results_file = copy.deepcopy(data_list[nn]['results_file'])
	cnums = copy.deepcopy(data_list[nn]['cnums'])
	drawregionsfile = r'E:\SAPMresults_Dec2022\SAPM_network_plotting_definition_2L_July2023.xlsx'
	sheetname = 'DBsig'
	regionnames = 'regions'
	statname = 'DB'
	figurenumber = 200
	scalefactor = 'auto'
	threshold_text = 'abs>0'
	writefigure = True

	regions = pysapm.define_drawing_regions_from_file(drawregionsfile)
	multiple_output = False
	if multiple_output:
		outputname = pysapm.draw_sapm_plot(results_file, sheetname, regionnames, regions, statname,
										   figurenumber, scalefactor, cnums, threshold_text, writefigure)
	else:
		outputname = pysapm.draw_sapm_plot_SO(results_file, sheetname, regionnames, regions, statname,
											  figurenumber, scalefactor, cnums, threshold_text, writefigure)
