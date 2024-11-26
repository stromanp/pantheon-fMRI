# files to read and analyze eye-tracking data in ascii format

import numpy as np
import matplotlib.pyplot as plt

fname = r'E:\FMstudy2023\MAR06_2023\eyetrackingdata\ps23_001_slow1\ps23_001.asc'
f = open(fname,'r')
data = f.read()

# find the flags to guide parsing the ascii data
flags = ['START\t', 'INPUT\t', 'END\t']
xlist = []
for n in range(len(flags)):
	if n == 0:
		startpoint = 0
	else:
		startpoint = xlist[n-1]
	l = len(flags[n])
	x = data[startpoint:].index(flags[n]) + startpoint
	xlist += [x]
	print('{}  {}'.format(x,data[x:x+l]))

# parse the data based on the flags

# starting point
tpos = xlist[1]
p = data[tpos:].index('\n') + tpos
text = data[tpos:p]
splittext = split_text_by_delimiter(text, '\t')
starttime = int(splittext[1])
rate = int(splittext[2])
tpos = p+1

messagelist = ['MSG','EFIX','SFIX','SSACC','ESACC']
eyedata = []
message_record = []
progress_message = np.zeros(10)
progress_count = 0
while tpos < xlist[-1]:
	p = data[tpos:].index('\n') + tpos
	text = data[tpos:p]
	splittext = split_text_by_delimiter(text, '\t')

	try:
		timestamp = int(splittext[0])
		xpos = float(splittext[1])
		ypos = float(splittext[2])
		pupilsize = float(splittext[3])
		datapoint = [timestamp, xpos, ypos, pupilsize]
		eyedata += [datapoint]
	except:
		# expect a flag
		for mm, msg in enumerate(messagelist):
			ml = len(msg)
			check = splittext[0][:ml] == msg
			if check:
				if msg == 'MSG':
					timestamp = splittext[1]
				else:
					timestamp = -1
				entry = {'msg':msg, 'text':text, 'index':tpos, 'timestamp':timestamp}
				message_record.append(entry)
	tpos = p+1

	progress = 100.0*(tpos-xlist[1])/(xlist[-1]-xlist[1])
	if progress > 10.0*progress_count:
		print('{:.1f} percent done ...'.format(progress))
		progress_count += 1

print('    done.')
eyedata = np.array(eyedata)

tt = eyedata[:,0]-eyedata[0,0]
windownum = 10
plt.close(windownum)
plt.figure(windownum)
plt.plot(tt,eyedata[:,3])

plt.close(windownum+1)
plt.figure(windownum+1)
plt.plot(tt,eyedata[:,1],'-r')
plt.plot(tt,eyedata[:,2],'-b')



def split_text_by_delimiter(text, delimiter):
	parsedtext = []
	keeplooking = True
	dd = len(delimiter)
	while keeplooking:
		try:
			c = text.index(delimiter)
			oneword = text[:c]
			parsedtext += [oneword]
			text = text[(c+dd):]
		except:
			keeplooking = False
			parsedtext += [text]
	return np.array(parsedtext)