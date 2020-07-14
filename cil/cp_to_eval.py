import os
os.makedirs("eval", exist_ok=True)
os.makedirs("eval/images", exist_ok=True)
os.makedirs("eval/groundtruth", exist_ok=True)
with open("val.txt", 'r') as f:
    files = f.readlines()

file_names = []
for item in files:
    item = item.strip()
    item = item.split('\t')
    img_name = item[0]
    gt_name = item[1]
    os.system("cp {} {}".format(img_name, img_name.replace("training", "eval")))
    os.system("cp {} {}".format(gt_name, gt_name.replace("training", "eval")))