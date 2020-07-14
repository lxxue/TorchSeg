import os

epoch_nums = ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500', '550']
for e in epoch_nums:
    os.system("python eval.py -e {} -p results_eval/epoch{}/".format(e, e))