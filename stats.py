#! /usr/bin/python3

import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data_file = "training_results.txt"

annealing_data = []
epoch_data = []
accuracy_data = []

treatment_1 = []
treatment_2 = []
treatment_3 = []
treatment_4 = []
treatment_5 = []
treatment_6 = []

with open(data_file) as file:
    for i, line in enumerate(file):
    	if i == 0:
    		continue
    	print(line.rstrip())
    	depth, width, epoch, annealing, accuracy = line.strip().split(',')
        
    	if annealing == 'True':
    		annealing_str = 'Annealing'
    	elif annealing == 'False':
    		annealing_str = 'Standard'

    	if epoch == '16':
    		epoch_str = 'Low'
    	elif epoch == '24':
    		epoch_str = 'Medium'
    	elif epoch == '32':
    		epoch_str = 'High'

    	annealing_data.append(annealing_str)
    	epoch_data.append(epoch_str)
    	accuracy_data.append(float(accuracy))

    	if epoch == '16' and annealing == 'True':
    		treatment_1.append(float(accuracy))
    	elif epoch == '24' and annealing == 'True':
    		treatment_2.append(float(accuracy))
    	elif epoch == '32' and annealing == 'True':
    		treatment_3.append(float(accuracy))
    	elif epoch == '16' and annealing == 'False':
    		treatment_4.append(float(accuracy))
    	elif epoch == '24' and annealing == 'False':
    		treatment_5.append(float(accuracy))
    	elif epoch == '32' and annealing == 'False':
    		treatment_6.append(float(accuracy))

graph_data = [treatment_1, treatment_2, treatment_3, treatment_4, treatment_5, treatment_6]
fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
bp = ax.boxplot(graph_data, #patch_artist = True,
                 vert = 0)
ax.set_yticklabels(['Annealing, Low Epoch', 'Annealing, Medium Epoch', 'Annealing, High Epoch', 
                    'Standard, Low Epoch', 'Standard, Medium Epoch', 'Standard, High Epoch' ])
plt.title("Training Results")
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.show()

def output_summary_statistics(dataset, label):
	print(f"Summary statistics for: {label}")
	print(f"\t Mean: {round(statistics.mean(dataset), 6)}")
	print(f"\t Medium: {round(statistics.median(dataset), 6)}")
	print(f"\t Standard Deviation: {round(statistics.stdev(dataset), 6)}")
	print(f"\t Sample Size: {len(dataset)}")


output_summary_statistics(treatment_1, 'Annealing, Low Epoch')
output_summary_statistics(treatment_2, 'Annealing, Medium Epoch')
output_summary_statistics(treatment_3, 'Annealing, High Epoch')
output_summary_statistics(treatment_4, 'Standard, Low Epoch')
output_summary_statistics(treatment_5, 'Standard, Medium Epoch')
output_summary_statistics(treatment_6, 'Standard, High Epoch')


dataframe = pd.DataFrame({'Annealing': annealing_data,
                   'Epochs': epoch_data,
                   'Accuracy': accuracy_data})

print(dataframe)
model = ols('Accuracy ~ C(Annealing) + C(Epochs) + C(Annealing):C(Epochs)', data=dataframe).fit()
print(sm.stats.anova_lm(model, typ=2))


# Perform Tukey's HSD test for Factor1
tukey_results = pairwise_tukeyhsd(dataframe['Accuracy'], dataframe['Annealing'], alpha=0.05)

# Print the results
print(tukey_results)