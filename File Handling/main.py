import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "file.txt")

f = open(file_path, "r")
data = f.read()
print(data)
f.close()
