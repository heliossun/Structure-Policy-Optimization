
import json
import random
random.seed(10)

question_file=""
out_path=""
sampleNum=200000

old_dt = json.load(open(question_file,'r'))
data_dict=[]
for d in old_dt:
    if 'video' in d:
        data_dict.append(d)
    elif 'image' in d:
        image_file = d["image"]
        if type(image_file) is list:
            if len(image_file) > 1:
                continue
            else:
                data_dict.append(d)
        else:
            data_dict.append(d)
del old_dt
print("Total image and video samples: ",len(data_dict))
if len(data_dict)>sampleNum:
    data_dict = random.sample(data_dict,sampleNum)
with open(out_path, "w",encoding='utf-8') as outfile:
    outfile.write(data_dict)