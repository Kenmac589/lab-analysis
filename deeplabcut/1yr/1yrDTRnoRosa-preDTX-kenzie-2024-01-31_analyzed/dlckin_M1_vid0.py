import dlc2kinematics as dlck

# %% [markdown]
# First we will load in the .h5 file of the desired video

# %%

df, bodyparts, scorer = dlck.load_data(
    './videos/1yrDTRnoRosa-M1-19102023_000000DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000.h5'
)

# %% [markdown]
# Lets get an idea of the body parts we're working with if we havne't looked at the file in a bit

# %%
print(bodyparts)
print(df)

# %%
leg_labels = [11, 10, 9, 8, 7, 6]
mirror_lables = [17, 16, 15, 14, 13]
leg_parts = [bodyparts[index] for index in leg_labels]
mirror_parts = [bodyparts[index] for index in mirror_lables]
print("Leg Labels")
print(leg_parts)
print()
print("Mirror Labels")
print(mirror_parts)

# %%
# Compute velocity and acceleration of the leg

leg_vel = dlck.compute_velocity(df, bodyparts=leg_parts)
leg_acc = dlck.compute_acceleration(df, bodyparts=leg_parts)
print(leg_vel)

# %%


