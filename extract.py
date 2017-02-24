
import numpy as np 
import h5py 
import os

files = os.listdir("./features_beverage_new")
#print files

j = 0

file1 = open("beverage_train_new.txt","w")
file2 = open("beverage_predict_new.txt","w")

for filename in files:
    if filename.endswith(".jpg.h5"):
        filename = "./features_beverage_new" + "/" + filename
        #print filename

        print filename
        f = h5py.File(filename,"r")
        
        
        
        a =  f[u'feat'].value
        line = "0" + "\t" + "1"
        for i in a:
            line = line + "\t" + str(i)
            
        if j <= 60:

            file1.write(line)
            file1.write("\n")
        elif(j<= 84):
            line = filename + "\t" + line
            file2.write(line)
            file2.write("\n")
        #print line
        else:
            print "xxxxxx"
            
        j = j + 1
         
      
        print j

    else:
        continue

file1.close()
file2.close()








