with open("train.txt", 'r') as fin:
    with open("train_edge_midline.txt", 'w') as fout:
        files = fin.readlines()
        for item in files:
            item = item.strip()
            item = item.split('\t')
            img_name = item[0]
            gt_name = item[1]
            
            img_name_wo_dir = img_name.split('/')[-1]
            # prefix = '../../../cil/'
            prefix = ''
            edge_name = prefix+'zhang-suen-thinning/edges/'+img_name_wo_dir
            midline_name = prefix+'zhang-suen-thinning/midlines/'+img_name_wo_dir
            img_name = prefix + img_name
            gt_name = prefix + gt_name

            fout.write("{}\t{}\t{}\t{}\n".format(img_name, gt_name, edge_name, midline_name))

