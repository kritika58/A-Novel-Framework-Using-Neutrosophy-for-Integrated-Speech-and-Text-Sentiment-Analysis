import glob
import os
main_dir = "C:\\Users\\lenovo\\Desktop\\FINAL PROJECT\\LibriSpeech\\dev-clean\\"
sub_dirs=os.listdir(main_dir)  
print(sub_dirs)
file_ext="*.wav"

for sub_dir in enumerate(sub_dirs):
    print("sub_dir:", sub_dir[1])
    sub_dirs_2=os.listdir(str(main_dir+sub_dir[1]))
    for sd in enumerate(sub_dirs_2):
        for fn in glob.glob(os.path.join(main_dir, sub_dir[1], sd[1], file_ext)):
            print(fn)


#myFile=open ('Text File.txt', 'r')

#Printing the files original text first
#for line in myFile.readlines():
#print line

#Splitting the text
#varLine = line
#splitLine = varLine.split (". ") 

#Printing the edited text
#print splitLine

#Closing the file
#myFile.close()