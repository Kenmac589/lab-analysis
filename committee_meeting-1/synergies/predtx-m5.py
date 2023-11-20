import motorpyrimitives as mp

# Pre DTX group
data_selection_non, syn_selection_non = './full_width_test/norm-emg-preDTX-100.csv', 3
motor_p_non, motor_m_non = mp.synergy_extraction(data_selection_non, syn_selection_non)
fwhl_non, fwhl_non_start_stop, fwhl_height_non = mp.full_width_half_abs_min(motor_p_non, syn_selection_non)

data_selection_per, syn_selection_per = './full_width_test/norm-emg-preDTX-per.csv', 3
motor_p_per, motor_m_per = mp.synergy_extraction(data_selection_per, syn_selection_per)
fwhl_per, fwhl_per_start_stop, fwhl_height_per = mp.full_width_half_abs_min(motor_p_per, syn_selection_per)

mp.sel_primitive_trace(data_selection_non, syn_selection_non, "M5 PreDTX Non-pertubation 0.100m/s")
mp.sel_primitive_trace(data_selection_per, syn_selection_per, "M5 PreDTX with Pertubation 0.100m/s")
