# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:13:56 2022

@author: m.lambert
"""
import srs_engine as srs
import pandas as pd

# Build frequency array for shock response computation
# fn_array = np.array([30,85,250])
fn_array = srs.build_freq_array()


# Build dummy input pulse/s
A = 10
T = 0.01

# Half sine
x, y, name  = srs.create_half_sine(A=A, T=T, num_half_sins=1)

# Sawtooth
# x, y, name  = srs.create_sawtooth(A=A, T=T)

# Step input
# x, y, name  = srs.create_step_input(A=A, T=T)


# Put input pulse into a pd.Dataframe
df = pd.DataFrame({'time':x, 'data':y})
df = df.set_index('time')

# Send pandas Dataframe to the srs engine
test_srs = srs.SRS(df)
test_srs.name = '{}g {}ms {}'.format(A,T*1000,name)



# PLot shock response spectrum using both methods.

# Kelly & Richman
# test_srs.calc_kellyrichman_srs(fn_array=srs.build_freq_array())
# test_srs.plot_results()
# test_srs.calc_kellyrichman_srs(fn_array=fn_array)
# test_srs.plot_inout_data()

# Smallwood
# test_srs.calc_smallwood_srs(fn_array=srs.build_freq_array(start=1,end=1000,oct_steps=1))
# test_srs.plot_results()
test_srs.calc_smallwood_srs(fn_array=fn_array)
test_srs.plot_results()
test_srs.plot_inout_data()
# test_srs.plot_srs()
# test_srs.plot_pvss()
