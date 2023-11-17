import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Data

kinematicData = pd.read_csv('CoM-M3-WT-20220420-3-kinematics.csv')

# Data Preview
kinematicData.head()

# kinematicData.plot(x='Time', y=['CoMx', 'Hipx', 'Hipy', 'HRx', 'HRy', 'toex', 'toey'], kind='line')

kinData = np.array(kinematicData)

# Only needed if working on multiple recordings from varying start and stop points
# rangeBegin = 37359
# rangeEnd = 37735

# Channels used
# channels = ['36 CoMx (cm)', '15 Hipx (cm)', '16 Hipy (cm)', '29 HRy (cm)',  ] 

selectedRange = 363

# Making sure I'm not eyeballing indices
index = kinematicData.columns.get_loc('36 CoMx (cm)')
print("CoMx", index)
index = kinematicData.columns.get_loc('15 Hipx (cm)')
print("Hipx", index)
index = kinematicData.columns.get_loc('16 Hipy (cm)')
print("Hipy", index)
index = kinematicData.columns.get_loc('29 HRx (cm)')
print("HRx", index)
index = kinematicData.columns.get_loc('30 HRy (cm)')
print("HRy", index)
index = kinematicData.columns.get_loc('23 toex (cm)')
print("toex", index)
index = kinematicData.columns.get_loc('24 toey (cm)')
print("toey", index)
index = kinematicData.columns.get_loc('24 toey (cm)')
print("toey", index)
index = kinematicData.columns.get_loc('24 toey (cm)')
print("toey", index)

# Plot all of kinematicData against time
sns.set()
plt.plot(kinData[0:selectedRange,0], kinData[0:selectedRange,1], label='CoM')
plt.plot(kinData[0:selectedRange,0], kinData[0:selectedRange,2], label='Treadmill Speed')
plt.plot(kinData[0:selectedRange,0], kinData[0:selectedRange,3], label='CoMx')
plt.plot(kinData[0:selectedRange,0], kinData[0:selectedRange,4], label='Hipx')
plt.plot(kinData[0:selectedRange,0], kinData[0:selectedRange,5], label='Hipy')
plt.plot(kinData[0:selectedRange,0], kinData[0:selectedRange,6], label='HRx')
plt.plot(kinData[0:selectedRange,0], kinData[0:selectedRange,7], label='HRy')
plt.plot(kinData[0:selectedRange,0], kinData[0:selectedRange,8], label='toex')
plt.plot(kinData[0:selectedRange,0], kinData[0:selectedRange,9], label='toey')
plt.ylabel("Lateral Position")
plt.xlabel("Time(s)")
plt.legend(loc='lower right')
plt.show()

# Assign time along X axis
time = kinData[0:selectedRange,0]
treadmillSpeed = kinData[0:selectedRange,2]
CoM = kinData[0:selectedRange,26]
hipX= kinData[0:selectedRange,6]
hipY = kinData[0:selectedRange,7]
hindlimbRightX = kinData[0:selectedRange, 20]
hindlimbRightY = kinData[0:selectedRange, 21]
toe_y = kinData[0:selectedRange, 15]

print(CoM[:10])

# Correcting hindlimbRightY values with approximation of treadmill floor by ToeY values
floor_correction = 7.7733
hry_adjusted = hindlimbRightY - floor_correction

# correctedHRy = np.subtract(hindlimbRightY, floor_correction)
print('Orginal values for HRy:', hindlimbRightY[:10])
print('Transformed values', hry_adjusted[:10])

# Creating array for xCoM
xCoMOriginal = CoM + ((treadmillSpeed * CoM) / np.sqrt(9.81 / hindlimbRightY))
xCoM = CoM + ((treadmillSpeed * CoM) / np.sqrt(9.81 / hry_adjusted))

# Verifying
print(xCoMOriginal[:10])
print(xCoM[:10])

# Plotting
sns.set()
plt.plot(time, CoM, label='CoM')
plt.plot(time, xCoM, label='xCoM')
plt.plot(time, xCoMOriginal, label='Original Calculation')
# plt.plot(time, hipX, label="Right Hip X")
# plt.plot(time, hipY)
# plt.plot(time, hindlimbRightY, label="Hindlimb Y")
plt.ylabel("Lateral Position")
plt.xlabel("Time(s)")
plt.legend(loc='lower right')
# plt.plot(time, hindlimbRight)

plt.show()
