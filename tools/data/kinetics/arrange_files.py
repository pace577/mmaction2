import sys
import os
import shutil

try:
  dataset_name = sys.argv[1]
except IndexError as e:
  print(f"{e}. Provide dataset name (kinetics400 or kinetics600) as the argument to the script")
  exit()
if dataset_name in ["kinetics400", "kinetics600"]:
  os.chdir(f"../../../data/{dataset_name}/val")
else:
  raise FileNotFoundError("Argument to the script must be 'kinetics400' or 'kinetics600'")

with open("../annotations/kinetics_train.csv") as annotations:
  lines = annotations.readlines()
files_list = os.listdir()

for line in lines[1::]:
  split_line = line.split(",") #class name at idx=0, youtube id at idx 1
  class_name = split_line[0].replace(' ','_')
  #file_name = split_line[1]+"_"+("0"*(6-len(split_line[2])))+split_line[2]+"_"+("0"*(6-len(split_line[3])))+split_line[3]+".mp4"
  if os.path.isdir(class_name) is False:
    os.mkdir(class_name)
  try:
    file_name = None
    for item in files_list:
      if split_line[1] in item:
        file_name = item
        break
    shutil.move(file_name, os.path.join(class_name,file_name))
  except FileNotFoundError as e:
      print(e)
      print("File Not found, Youtube ID:", split_line[1])
  except TypeError as e:
      print(f"No such file with the YouTube ID {split_line[1]} is present")
