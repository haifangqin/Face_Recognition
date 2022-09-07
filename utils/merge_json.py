import json
import sys

file_names = sys.argv[1]
file_names = file_names.strip().split(',')
all_faces = []
for file_name in file_names:
    f = open(file_name)
    data = json.load(f)
    all_faces.extend(data)
print(len(all_faces))
out_file = open(sys.argv[2], 'w')
json.dump(all_faces, out_file, indent = 4)
out_file.close()
