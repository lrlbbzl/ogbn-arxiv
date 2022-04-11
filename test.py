import torch 

file_name = "cassette.txt"
with open(file_name, "a+") as fp:
    fp.write("±")
fp.close()