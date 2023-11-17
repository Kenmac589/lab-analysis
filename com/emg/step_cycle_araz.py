# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 17:07:20 2023

@author: Araz
"""


import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import f_oneway

# Read the data for group 1 and group 2
group1_data = pd.read_excel(r'D:\Thesis\Data for analysis\Duty cycle\Righ limb data\RH perturbation.xlsx')
group2_data = pd.read_excel(r'D:\Thesis\Data for analysis\EGR3\EGR3 Data\Perturbation\All Perturbation.xlsx')

# Iterate over different treadmill speeds
treadmill_speeds = group1_data['Speed'].unique()
treadmill_speeds = group2_data['Speed'].unique()

# Create an empty dictionary to store the results
results = {}

# Iterate over the treadmill speeds
for speed in treadmill_speeds:
    # Get the data for the current speed for group 1 and group 2
    group1_speed_data = group1_data[group1_data['Speed'] == speed]
    group2_speed_data = group2_data[group2_data['Speed'] == speed]

    # Calculate the means for CP, swing duration, and stance duration for group 1
    group1_RFX_mean = group1_speed_data['RFX'].mean()
    group1_RHX_mean = group1_speed_data['RHX'].mean()
    group1_distance_mean = (group1_speed_data['RFX'] - group1_speed_data['RHX']).mean()

    # Calculate the means for CP, swing duration, and stance duration for group 2
    group2_RFX_mean = group2_speed_data['RFX'].mean()
    group2_RHX_mean = group2_speed_data['RHX'].mean()
    group2_distance_mean = (group2_speed_data['RFX'] - group2_speed_data['RHX']).mean()

    # Calculate the standard deviations for CP, swing duration, and stance duration for group 1
    group1_RFX_error = group1_speed_data['RFX'].std()
    group1_RHX_error = group1_speed_data['RHX'].std()
    group1_distance_error = (group1_speed_data['RFX'] - group1_speed_data['RHX']).std()

    # Calculate the standard deviations for CP, swing duration, and stance duration for group 2
    group2_RFX_error = group2_speed_data['RFX'].std()
    group2_RHX_error = group2_speed_data['RHX'].std()
    group2_distance_error = (group2_speed_data['RFX'] - group2_speed_data['RHX']).std()

    ## Perform t-test between group 1 and group 2 for CP
    ttest_RFX = ttest_ind(group1_speed_data['RFX'], group2_speed_data['RFX'])

    # Perform t-test between group 1 and group 2 for swing duration
    ttest_RHX = ttest_ind(group1_speed_data['RHX'], group2_speed_data['RHX'])
    

    # Perform ANOVA between group 1 and group 2 for CP, swing duration, and stance duration
    # anova_cp = f_oneway(group1_speed_data['CP'], group2_speed_data['CP'])
    # anova_swing = f_oneway(group1_speed_data['Swing duration'], group2_speed_data['Swing duration'])
    # anova_stance = f_oneway(group1_speed_data['Stance duration'], group2_speed_data['Stance duration'])
    # anova_phase = f_oneway(group1_speed_data['RH/RF'], group2_speed_data['RH&RF'])

    # Store the results in the dictionary
    results[speed] = {
        'Group1 RFX Mean': group1_RFX_mean,
        'Group2 RFX Mean': group2_RFX_mean,
        'Group1 RHX Mean': group1_RHX_mean,
        'Group2 RHX Mean': group2_RHX_mean,
        'Group1 Distance Mean': group1_distance_mean,
        'Group2 Distance Mean': group2_distance_mean,
        'Group1 RFX Error': group1_RFX_error,
        'Group1 RHX Error': group1_RHX_error,
        'Group2 RFX Error': group2_RFX_error,
        'Group2 RHX Error': group2_RHX_error,
        'Group1 Distance error': group1_distance_error,
        'Group2 Distance error': group2_distance_error,
        #'Group1 phase Mean': group1_phase_mean,
        #'Group2 phase Mean': group2_phase_mean,
        'RFX T-Test': ttest_RFX,
        'RHX T-Test': ttest_RHX,
        #'Distance T-Test': ttest_distance,
        #'phase T-Test': ttest_phase,
        #'CP ANOVA': anova_cp,
        #'Swing ANOVA': anova_swing,
        #'Stance ANOVA': anova_stance,
        #'phase ANOVA': anova_phase,
    }

# Create a pandas DataFrame from the results dictionary
results_df = pd.DataFrame.from_dict(results, orient='index')

# Write the results to an Excel file
results_df.to_excel('resultss.xlsx', index=True)
