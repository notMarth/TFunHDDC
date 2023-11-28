import csv
import sys
sys.path.append("../../..")
import TFunHDDC.NOxBenchmark as NOx

classes = NOx.fitNOxBenchmark()['target']
classes = classes.astype(int)

filename = 'classes.csv'
with open(filename, 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile)
    datawriter.writerow(classes)