import os

def make_unique_directory_path(path):
	unique_path=os.path.normpath(path)
	base_name=os.path.basename(unique_path)
	base_path=os.path.dirname(unique_path)
	file_num=2
	while os.path.exists(unique_path):
		name=base_name+"_"+str(file_num)
		file_num=file_num+1
		unique_path=base_path+"/"+name
	return unique_path+"/"