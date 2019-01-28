"""
ENGR-E 599: Psychophysiological Engineering

Polygraph Project

@author: Vedang Narain

2 November 2018
"""

#==============================================================================
#-------------------------------------NOTES------------------------------------
#==============================================================================

"""
— This code evaluates one subject at a time.

– To change the subject, make edits to lines 188 and 189 (necessary), and lines
  254, 316, 317, and 318 (to change graph titles).

– To change the period of interest for quick viewing, make changes to lines 225
  and 226.

– To change the sampling rate, change line 199.

– To evaluate a particular question, use the functions.

— Acronyms: RIQ = Recent Irrelevant Questions
            IPP = Immediately Preceding Period
            HR = Heart Rate

— For best results, please view the graphs in the IPython console or the report.
"""

#==============================================================================
#-----------------------------------LIBRARIES----------------------------------
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

#==============================================================================
#-----------------------------------FUNCTIONS----------------------------------
#==============================================================================

"""
Note: The functions below only accept inputs for relevant (non-control) questions.
      In other words, questions 1A, 2A, 7A, and 8A are used as baseline regions
      since they are definite lies, and should not be treated as relevant
      questions. Question 7D has been considered an irrelevant question as well.
      It is only used to mark the end of the response time for the preceding
      question.
"""

# define function for Butterworth Low-pass Filter
def butterworth_lpf(unfiltered_signal, cutoff_freq, order):
    nyq_freq = 0.5 * Fs
    normalized_cutoff = cutoff_freq/nyq_freq
    b, a = signal.butter(order, normalized_cutoff)
    filtered_signal = signal.filtfilt(b, a, unfiltered_signal)
    return filtered_signal


# define function to add event separators
def separators():
    for i in event_times:
        if i < end_time:
            plt.axvline(x = i, color = 'red', alpha = 0.5)


# define function to calculate simple 5-second IPP baseline (unused)
def simple_ipp_baseline(parameter, event_row):  # accepts (full_signal, row_number_of_question_in_events_file)
    event_index = int(Fs * event_times[event_row - 2])
    return np.mean(parameter[event_index - (4 * Fs) : event_index + 1])


# define function to generate start and end indices for RIQ baseline
def get_riq_indices(event_row):  # accepts (row_of_question_in_events_file)
    if 6 < event_row < 11:
        start_index = int(Fs * event_times[3])
        end_index = int(Fs * event_times[5])
    elif 12 < event_row < 38:
        start_index = int(Fs * event_times[9])
        end_index = int(Fs * event_times[11])
    return start_index, end_index


# define function to calculate simple RIQ baseline (use only for EDA)
def riq_eda_baseline(event_row):  # accepts (row_of_question_in_events_file)
    start_index, end_index = get_riq_indices(event_row)
    return np.mean(full_skin[start_index : end_index])


# define function to calculate RIQ systolic baseline (use only for BP)
def riq_systolic_baseline(event_row):  # accepts (row_of_question_in_events_file)
    systolic_list_bp = []
    start_index, end_index = get_riq_indices(event_row)
    for i in peak_pos_bp:
        if start_index <= i <= end_index:
            systolic_bp = full_bp[i]
            systolic_list_bp.append(systolic_bp)
    return np.mean(systolic_list_bp)


# define function to calculate RIQ IBI baseline (use only for Pulse (PPG))
def riq_ibi_baseline(event_row):  # accepts (row_of_question_in_events_file)
    systolic_list_ppg = []
    interval_list_ppg = []
    cnt = 0
    start_index, end_index = get_riq_indices(event_row)
    for i in peak_pos_ppg:
        if start_index <= i <= end_index:
            systolic_list_ppg.append(i)
    while (cnt < len(systolic_list_ppg) - 1):
        interval = ((systolic_list_ppg[cnt + 1] - systolic_list_ppg[cnt])) / Fs
        interval_list_ppg.append(interval)
        cnt += 1
    return np.mean(interval_list_ppg)


# define function to generate start and end indices for question segment of interest
def get_question_indices(event_row):  # accepts (row_of_question_in_events_file)
    start_index = int(Fs * event_times[event_row - 2])
    end_index = int(Fs * event_times[event_row - 1])
    return start_index, end_index


# define function to calculate simple average (use only for EDA)
def average_eda(event_row):  # accepts (row_of_question_in_events_file)
    start_index, end_index = get_question_indices(event_row)
    return np.mean(full_skin[start_index : end_index])


# define function to calculate systolic average (use only for BP)
def average_systolic(event_row):  # accepts (row_of_question_in_events_file)
    systolic_list_bp = []
    start_index, end_index = get_question_indices(event_row)
    for i in peak_pos_bp:
        if start_index <= i <= end_index:
            systolic_bp = full_bp[i]
            systolic_list_bp.append(systolic_bp)
    return np.mean(systolic_list_bp)


# define function to calculate average IBI (use only for Pulse (PPG))
def average_ibi(event_row):  # accepts (row_of_question_in_events_file)
    systolic_list_ppg = []
    interval_list_ppg = []
    cnt = 0
    start_index, end_index = get_question_indices(event_row)
    for i in peak_pos_ppg:
        if start_index <= i <= end_index:
            systolic_list_ppg.append(i)
    while (cnt < len(systolic_list_ppg) - 1):
        interval = ((systolic_list_ppg[cnt + 1] - systolic_list_ppg[cnt])) / Fs
        interval_list_ppg.append(interval)
        cnt += 1
    return np.mean(interval_list_ppg)


# define function to calculate and quantify percentage change from RIQ baseline
def baseline_change(event_row):  # accepts (row_of_question_in_events_file)
    eda_change = ((average_eda(event_row) / riq_eda_baseline(event_row)) * 100) - 100
    systolic_change = ((average_systolic(event_row) / riq_systolic_baseline(event_row)) * 100) - 100
    ibi_change = ((average_ibi(event_row) / riq_ibi_baseline(event_row)) * 100) - 100
    return (eda_change, systolic_change, ibi_change)


# define function to plot bar chart of responses relative to RIQ baseline
def bar_chart(plot_title, change_list):  # e.g. ('Subject A: EDA', eda_change_list)
    plt.figure()
    plt.rc('xtick', labelsize = 7)
    plt.bar(question_positions, change_list, align = 'center')
    plt.xticks(question_positions, question_names)
    plt.title(plot_title)
    plt.xlabel('Questions')
    plt.ylabel('Average Change (%)')
    plt.grid(True, alpha = 0.5)
    plt.show()

#==============================================================================
#----------------------------IMPORT AND PREPARE DATA---------------------------
#==============================================================================

# import data for subject of choice
raw_data = pd.read_table('Subject G.txt', skiprows = 15)
raw_events = pd.read_excel('SubjectG_events.xlsx', skiprows = [38])

# remove extra row while retaining header
corrected_data = raw_data.reindex(raw_data.index.drop(0))

# isolate individual arrays from event table
event_times = raw_events.iloc[:, 0]
event_labels = raw_events.iloc[:, 1]

# choose sampling rate (from 10 Hz up to 1 kHz)
Fs = 10

# calculate downsampling factor
downsampling_factor = 1000/Fs

# import raw signals into individual arrays
raw_bp = corrected_data.iloc[:, 1].values  # Raw Blood Pressure (mmHg)
raw_rsp = corrected_data.iloc[:, 2].values  # RSP (V)
raw_ppg = corrected_data.iloc[:, 3].values  # PPG (V)
# raw_arterial = corrected_data.iloc[:, 4].values  # Mean Arterial Pressure (mmHg)
raw_bpm = corrected_data.iloc[:, 5].values  # Pulse (BPM)
raw_skin = corrected_data.iloc[:, 6].values  # EDA (mS)

# downsample all signals to chosen sampling frequency
full_bp = raw_bp[::int(downsampling_factor)]
full_rsp = raw_rsp[::int(downsampling_factor)]
full_ppg = raw_ppg[::int(downsampling_factor)]
# full_arterial = raw_arterial[::int(downsampling_factor)]
full_bpm = raw_bpm[::int(downsampling_factor)]
full_skin = raw_skin[::int(downsampling_factor)]

#==============================================================================
#--------------------------CROP SIGNAL FOR EXAMINATION-------------------------
#==============================================================================

# choose start and end times of segment to be analyzed (in seconds)
start_time = 0
end_time = 500

# convert minutes to samples
start_sample = start_time * Fs
end_sample = end_time * Fs

# crop segments of signals for graphs
bp = full_bp[start_sample:end_sample]  # Raw Blood Pressure (mmHg)
rsp = full_rsp[start_sample:end_sample]  # RSP (V)
ppg = full_ppg[start_sample:end_sample]  # PPG (V)
# arterial = full_arterial[start_sample:end_sample]  # Mean Arterial Pressure (mmHg)
bpm = full_bpm[start_sample:end_sample]  # Pulse (BPM)
skin = full_skin[start_sample:end_sample]  # EDA (mS)

# obtain number of data points
length = len(bp)

# prepare time array
time_axis = np.linspace(start_time, end_time, length)

#==============================================================================
#--------------PLOT PARAMETER SNAPSHOTS FOR CHOSEN TIME INTERVAL---------------
#==============================================================================

# plot parameter values for chosen time interval
plt.figure()
plt.subplot(5, 1, 1)
plt.plot(time_axis, bp)
plt.title('Subject G')
plt.xlim(start_time, end_time)
# plt.ylim(40, 100)  # Line can be unmuted for Subject A to exclude error signal from graph.
separators()
plt.ylabel('BP (mmHg)')
plt.subplot(5, 1, 2)
plt.plot(time_axis, rsp)
plt.xlim(start_time, end_time)
separators()
plt.ylabel('RSP (V)')
plt.subplot(5, 1, 3)
plt.plot(time_axis, ppg)
plt.xlim(start_time, end_time)
separators()
plt.ylabel('Pulse (V)')
plt.subplot(5, 1, 4)
plt.plot(time_axis, skin)
plt.xlim(start_time, end_time)
separators()
plt.ylabel('EDA (mS)')
plt.subplot(5, 1, 5)
plt.plot(time_axis, bpm)
plt.xlim(start_time, end_time)
separators()
plt.xlabel('Time (s)')
plt.ylabel('HR (bpm)')
plt.subplots_adjust(top = 3)
plt.show()

#==============================================================================
#--------------------FIND PEAKS FOR PULSE AND BLOOD PRESSURE-------------------
#==============================================================================

# generate peak positions for blood pressure and pulse (0.5-second gap allows for HR of 120 bpm)
peak_pos_bp, _ = signal.find_peaks(full_bp, prominence = 5, distance = 0.5 * Fs)
peak_pos_ppg, _ = signal.find_peaks(full_ppg, distance = 0.5 * Fs)

#==============================================================================
#-----------------------------QUANTIFY RESPONSES-------------------------------
#==============================================================================

# initialize lists for lists of changes
eda_change_list = []
systolic_change_list = []
ibi_change_list = []

# generate lists of changes
for i in range(7, 38):
    if i in {11, 12}:
        continue
    eda_change, systolic_change, ibi_change = baseline_change(i)
    eda_change_list.append(eda_change)
    systolic_change_list.append(systolic_change)
    ibi_change_list.append(ibi_change)

# create list of question names and positions for bar charts
question_names = ('3A', '4A', '5A', '6A', '9A', '1B', '2B', '3B', '4B', '5B',
                  '6B', '7B', '8B', '9B', '1C', '2C', '3C', '4C', '5C', '6C',
                  '7C', '8C', '9C', '1D', '2D', '3D', '4D', '5D', '6D')
question_positions = np.arange(len(question_names))

# plot charts for EDA, BP, and HR changes
bar_chart('Subject G: Electrodermal Activity', eda_change_list)
bar_chart('Subject G: Systolic Pressure', systolic_change_list)
bar_chart('Subject G: Inter-beat Interval', ibi_change_list)
