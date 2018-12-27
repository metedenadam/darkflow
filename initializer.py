import os
import cv2
from sys import argv
from dlgdrive import download_file_from_google_drive

args = argv

originclasscount = int(args[1])
originfiltercount = (originclasscount + 5) * 5

ClassesDict = {}
logolist = []
logospath = 'FlickrLogos-v2'
filespath = 'logos/images'
annpath = 'logos/annotations'
testfilespath = 'logos_test/images'
testannpath = 'logos_test/annotations'
valfilespath = 'logos_val/images'
valannpath = 'logos_val/annotations'

origincfg = args[2]
configcfg = args[3]
w_gdriveid = args[4]
w_filedest = args[5]
download = True if args[6] == "True" else False
argc = len(args)
for i in range(7, argc):
	logolist.append(args[i])
	ClassesDict[args[i]] = i-7 

classcount = len(logolist)
filtercount = (classcount + 5) * 5

if classcount == 0:
	with open('all.spaces.txt', 'r') as all_spaces:
		for line in all_spaces:
			line = line.split()
			if line[0] == 'no-logo':
				break
			logolist.append(line[0])
	logolist = list(set(logolist))
	classcount = len(logolist)



xmlBeginTemp = """
<annotation>
	<folder>obj</folder>
	<filename>%(filename)s</filename>
	<source>
		<database>FlickrLogos-32_dataset_v2</database>
		<annotation>FlickrLogos-32</annotation>
		<image>flickr</image>
		<flickrid>0</flickrid>
	</source>
	<owner>
		<flickrid>noname</flickrid>
		<name>noname</name>
	</owner>
	<size>
		<width>%(width)d</width>
		<height>%(height)d</height>
		<depth>%(depth)d</depth>
	</size>
	<segmented>0</segmented>"""
objTemp = """
	<object>
		<name>%(name)s</name>
		<pose>Left</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>%(xmin)s</xmin>
			<ymin>%(ymin)s</ymin>
			<xmax>%(xmax)s</xmax>
			<ymax>%(ymax)s</ymax>
		</bndbox>
	</object>"""
xmlEndTemp = """
</annotation>"""

def download_dataset():
	if not os.path.exists(logospath):
		print(logospath + ' folder not found...')
		if not os.path.isfile('FlickrLogos-32_dataset_v2.zip'):
			print('FlickrLogos-32 dataset file not found, downloading...')
			os.system('wget http://www.multimedia-computing.de/flickrlogos/data/FlickrLogos-32_dataset_v2.zip')
		print('FlickrLogos-32 dataset file downloaded, unzipping...')
		os.system('unzip FlickrLogos-32_dataset_v2.zip')
	if os.path.exists(logospath):
		print('FlickrLogos-v2 folder found! Fixing HP...')
		os.system('mv FlickrLogos-v2/classes/masks/hp FlickrLogos-v2/classes/masks/HP')
	else:
		print('Unzip failed')
	print('Images downloaded and unpacked')

def configure_darkflow(is_new = True):
	os.system('rm ' + configcfg)
	os.system('cp ' + origincfg + ' ' + configcfg)
	print(configcfg + ' file created as a copy of ' + origincfg)
	if w_gdriveid != '':
		download_file_from_google_drive(w_gdriveid, w_filedest)
		print(w_filedest + ' file downloaded from Google Drive...')	
	with open(configcfg, 'r') as s:
		lines = s.read()
		lines = lines.replace('classes=' + str(originclasscount), 'classes='+str(classcount))
		lines = lines.replace('filters=' + str(originfiltercount),'filters='+str(filtercount))
		f = open(configcfg, 'w')
		f.write(lines)
		f.close()
		print(configcfg + ' modified accordingly...')
	if is_new:	
		with open('labels.txt', 'w') as labels:
			labels.write('\n'.join(logolist))
			print('labels.txt file created accordingly...')

def create_directories():
	try:
		os.mkdir('bin')
	except FileExistsError:
		print('bin already exists')

	try:
		os.mkdir('logos')
	except FileExistsError:
		print('logos already exists')

	try:
		os.mkdir('logos_test')
	except FileExistsError:
		print('logos_test already exists')

	try:
		os.mkdir('logos_val')
	except FileExistsError:
		print('logos_test already exists')

	try:
		os.mkdir(filespath)
	except FileExistsError:
		print('logos/images already exists')

	try:
		os.mkdir(annpath)
	except FileExistsError:
		print('logos/annotations already exists')

	try:
		os.mkdir(testfilespath)
	except FileExistsError:
		print('logos_test/images already exists')

	try:
		os.mkdir(testannpath)
	except FileExistsError:
		print('logos_test/annotations already exists')

	try:
		os.mkdir(valfilespath)
	except FileExistsError:
		print('logos_val/images already exists')

	try:
		os.mkdir(valannpath)
	except FileExistsError:
		print('logos_val/annotations already exists')

def convert(size, box):
   x = (box[0] + box[1])/2.0
   y = (box[2] + box[3])/2.0
   w = box[1] - box[0]
   h = box[3] - box[2]
   dw = 1./size[0]
   dh = 1./size[1]
   x = x*dw
   w = w*dw
   y = y*dh
   h = h*dh
   return (x,y,w,h)

def move_files_with_bboxes_2():
	os.system('mv FlickrLogos-v2/classes/masks/hp FlickrLogos-v2/classes/masks/HP')
	with open(logospath + '/all.spaces.txt', 'r') as all_spaces:
		for idx, line in enumerate(all_spaces):
			line = line.split()
			if line[0] == "no-logo":
				continue
			elif line[0] in logolist:
				imgpath = logospath + '/classes/jpg/' + line[0] + '/' + line[1]
				img_h, img_w, img_d = cv2.imread(imgpath).shape
				if idx % 15 == 0:
					os.rename(imgpath, testfilespath + '/' + line[1])
				elif idx % 49 == 0:
					os.rename(imgpath, valfilespath + '/' + line[1])
				else:
					os.rename(imgpath, filespath + '/' + line[1])
				fbboxes = logospath + '/classes/masks/' + line[0] + '/' + line[1] + '.bboxes.txt'
				with open(fbboxes, 'r') as bboxes:
					new_text = ""
					bboxes.readline()
					bbox_list = bboxes.read()
					bbox_list = bbox_list.split('\n')
					bbox_list.pop()
				for bb in bbox_list:
					bb = list(map(int, bb.split()))
					x_min = bb[0]
					y_min = bb[1]
					x_max = x_min + bb[2]
					y_max = y_min + bb[3]
					classname = line[0]
					class_id = ClassesDict[classname]
					b = (float(x_min), float(x_max), float(y_min), float(y_max))
					bbox = convert((img_w,img_h), b)
					new_text += (str(class_id) + " " + " ".join([str(a) for a in bbox]) + '\n')
				if idx % 15 == 0:
					txt_path = testannpath + '/' + line[1][:-4] + '.txt'
				elif idx % 49 == 0:
					txt_path = valannpath + '/' + line[1][:-4] + '.txt'
				else:
					txt_path = annpath + '/' + line[1][:-4] + '.txt'
				with open(txt_path, 'w') as f:
					f.write(xmlcontent)

	os.system('rm -rf FlickrLogos-v2')
	print('Image files moved, xml files created, rest of FlickrLogos-v2 deleted...')


def move_files_with_bboxes():
	os.system('mv FlickrLogos-v2/classes/masks/hp FlickrLogos-v2/classes/masks/HP')
	create_directories()
	nologocount = 0
	os.system('cp ' + logospath + '/all.spaces.txt all.spaces.txt')
	with open(logospath + '/all.spaces.txt', 'r') as all_spaces:
		for idx, line in enumerate(all_spaces):
			if nologocount < 10 * classcount:
				line = line.split()
				if line[0] in logolist:
					imgpath = logospath + '/classes/jpg/' + line[0] + '/' + line[1]
					img_h, img_w, img_d = cv2.imread(imgpath).shape
					xmlbegindata = {'filename':line[1], 'width':img_w, 'height':img_h, 'depth':img_d}
					xmlcontent = xmlBeginTemp % xmlbegindata
					if idx % 15 == 0:
						os.rename(imgpath, testfilespath + '/' + line[1])
					elif idx % 49 == 0:
						os.rename(imgpath, valfilespath + '/' + line[1])
					else:
						os.rename(imgpath, filespath + '/' + line[1])
					
					if line[0] == 'no-logo':
						xmlcontent += xmlEndTemp
						if idx % 15 == 0:
							txt_path = testannpath + '/' + line[1][:-4] + '.xml'
						elif idx % 49 == 0:
							txt_path = valannpath + '/' + line[1][:-4] + '.xml'
						else:
							txt_path = annpath + '/' + line[1][:-4] + '.xml'
						with open(txt_path, 'w') as f:
							f.write(xmlcontent)
						nologocount += 1
					else:
						fbboxes = logospath + '/classes/masks/' + line[0] + '/' + line[1] + '.bboxes.txt'
						with open(fbboxes, 'r') as bboxes:
							bboxes.readline()
							bbox_list = bboxes.read()
							bbox_list = bbox_list.split('\n')
							bbox_list.pop()
						for bb in bbox_list:
							bb = list(map(int, bb.split()))
							x_min = bb[0]
							y_min = bb[1]
							x_max = x_min + bb[2]
							y_max = y_min + bb[3]
							classname = line[0]
							objdata = {'name':classname, 'xmin':x_min, 'ymin':y_min, 'xmax':x_max, 'ymax':y_max}
							objcontent = objTemp % objdata
							xmlcontent += objcontent
						xmlcontent += xmlEndTemp
						if idx % 15 == 0:
							txt_path = testannpath + '/' + line[1][:-4] + '.xml'
						elif idx % 49 == 0:
							txt_path = valannpath + '/' + line[1][:-4] + '.xml'
						else:
							txt_path = annpath + '/' + line[1][:-4] + '.xml'
						with open(txt_path, 'w') as f:
							f.write(xmlcontent)
			else:
				print("Reached no-logo limit.")
				break
	os.system('rm -rf FlickrLogos-v2')
	print('Image files moved, xml files created, rest of FlickrLogos-v2 deleted...')

if download:
    download_dataset()
move_files_with_bboxes()
configure_darkflow()