import numpy, matplotlib.pyplot
from scipy.signal import stft, find_peaks
from openpyxl import Workbook
from openpyxl import load_workbook

excel_table = r"C:\Users\Asnie\Downloads\Dataset_848\6 February 2019 NG Homes Dataset\6 February 2019 NG Homes Dataset Table.xlsx"

workbook_excel = load_workbook(excel_table, data_only=True)

#Convert Excel table to Python table
table = []
for row in workbook_excel.active.iter_rows(values_only = True):
    table.append(list(row))

header = table[0]
data = table[1:]

table_dict = {}
for row in data:
    FileName, Activity, Subject, Repetition = row
    table_dict[FileName] = (Activity, Subject, Repetition)

#Radar parameters
nRangeBins = 128   #Number of samples per chirp / number of range bins
tChirp = 1e-3      #Chirp duration in seconds (1 ms)
fs = 1 / tChirp    #Slow-time sampling frequency (Hz), number of chirps per second, how many frames/samples happen per second
time_window = 256   #How many time frames/samples looked at at once (in 1 STFT window)
shared_frames = time_window * 0.75   #Number of shared/overlapping time frames in consecutive STFT windows

#Loop through all files in the Excel table
for FileName, (Activity, Subject, Repetition) in table_dict.items():

    file_name = f"{Activity} - Subject: {int(Subject)} Repetition: {int(Repetition)}"

    #Load raw radar data file
    data_file = rf"C:\Users\Asnie\Downloads\Dataset_848\6 February 2019 NG Homes Dataset\{FileName}"

    complexData = []
    with open(data_file, "r") as f:   #Open the IQ data file
        for _ in range(4):   #Skip first 4 header lines containing radar parameters
            next(f)
        for line in f:
            complexData.append(complex(line.replace('i', 'j')))   #Convert MATLAB-style complex string to Python complex type

    complexData = numpy.array(complexData)   #Store all IQ samples in a NumPy array

    #Reshape into chirps × fast-time
    nChirps = complexData.size // nRangeBins
    iq = complexData[:nChirps * nRangeBins]   #Truncate to full number of chirps
    iq = iq.reshape(nChirps, nRangeBins)     #Reshape 1D array into 2D array: rows=chirps, columns=fast-time samples

    #Range FFT (fast-time)
    range_fft = numpy.fft.fft(iq, axis = 1)   #Compute FFT along fast-time (range) dimension
    range_fft = range_fft[:, :nRangeBins // 2]   #Keep only positive range bins
    range_fft -= numpy.mean(range_fft, axis = 0, keepdims = True)   #Remove static clutter / basic MTI

    signal = numpy.mean(range_fft, axis = 1)   
    #signal: 1D array of complex values over slow-time (chirps), averaged across all range bins

    #Doppler-Time STFT
    f, t, STFT = stft(signal, fs = fs, nperseg = time_window, noverlap = shared_frames, return_onesided = False)
    #Zxx: complex STFT result, rows = Doppler frequency bins, columns = time windows
    doppler_spectrogram = 10 * numpy.log10(numpy.abs(STFT))  
    #Convert magnitude of STFT to dB scale

    doppler_spectrogram = numpy.fft.fftshift(doppler_spectrogram, axes = 0)   #Shift zero Doppler frequency to center
    f = numpy.fft.fftshift(f)   #Shift frequency vector accordingly

    #Plot Doppler Frequency – Time Spectrogram
    matplotlib.pyplot.figure()
    matplotlib.pyplot.imshow(doppler_spectrogram, aspect = "auto", origin="lower", extent=[t[0], t[-1], f[0], f[-1]], cmap="jet")
    matplotlib.pyplot.xlabel("Time (s)", fontsize = 16)
    matplotlib.pyplot.ylabel("Doppler Frequency (Hz)", fontsize = 16)
    matplotlib.pyplot.title(f"Doppler – Time Spectrogram:\n{file_name}", fontsize = 18)
    cbar = matplotlib.pyplot.colorbar()
    cbar.ax.tick_params(labelsize = 14)
    cbar.set_label("Signal Strength (dB)", fontsize = 16)
    matplotlib.pyplot.tick_params(axis = "both", labelsize = 14)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(rf"C:\Users\Asnie\Downloads\Dataset_848\6 February 2019 NG Homes Dataset\Doppler-Time Spectrogram\Doppler-Time Spectrogram - {FileName}.png")
    matplotlib.pyplot.close()

    #Plain spectrograms for ML modelling (no title, axis, labelling)
    #Only save spectrograms for selected activities
    activity_classes = ["Drink water", "Fall", "Pick up object", "Sit down", "Stand up", "Walk"]

    if Activity in activity_classes:

        matplotlib.pyplot.figure()
        matplotlib.pyplot.imshow(doppler_spectrogram, aspect = "auto", origin = "lower", cmap = "jet")
        matplotlib.pyplot.axis("off")
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig(rf"C:\Users\Asnie\Downloads\Dataset_848\6 February 2019 NG Homes Dataset\{Activity}\{FileName} - {Activity}.png")
        matplotlib.pyplot.close()


    min_doppler_frequency = []
    max_doppler_frequency = []
    doppler_bandwidth = []
    peak_signal_strength = []
    standard_deviation_signal = []

    for i in range(doppler_spectrogram.shape[1]):   #Loop over time windows
        column = doppler_spectrogram[:, i]
        dominant_signal = column[column > (column.min() + 0.8 * (column.max() - column.min()))]
        dominant_bins = f[column > (column.min() + 0.8 * (column.max() - column.min()))]
        min_doppler_frequency.append(dominant_bins.min())
        max_doppler_frequency.append(dominant_bins.max())
        doppler_bandwidth.append(max_doppler_frequency[-1] - min_doppler_frequency[-1])
        peak_signal_strength.append(dominant_signal.max())
        standard_deviation_signal.append(numpy.std(dominant_signal))

    #Micro-Doppler peaks
    peaks_min, _ = find_peaks(-numpy.asarray(min_doppler_frequency), prominence = 5, distance = 5)
    micro_doppler_peaks_min = []
    time_of_micro_doppler_peaks_min = []

    for i in peaks_min:
        if min_doppler_frequency[i] < 0.05:
            micro_doppler_peaks_min.append(min_doppler_frequency[i])
            time_of_micro_doppler_peaks_min.append(t[i])

    peaks_max, _ = find_peaks(max_doppler_frequency, prominence=5, distance=5)
    micro_doppler_peaks_max = []
    time_of_micro_doppler_peaks_max = []

    for i in peaks_max:
        if max_doppler_frequency[i] > 0.05:
            micro_doppler_peaks_max.append(max_doppler_frequency[i])
            time_of_micro_doppler_peaks_max.append(t[i])

    #Feature vector
    feature_vector = []
    features = [min_doppler_frequency, max_doppler_frequency, doppler_bandwidth, peak_signal_strength, standard_deviation_signal]
    for i in features:
        i = numpy.asarray(i)
        feature_vector.append(i.min())
        feature_vector.append(i.max())
        feature_vector.append(i.mean())
        feature_vector.append(i.std())

    workbook_features = Workbook()
    workbook_features.active.title = "Features"
    workbook_features.active.append(["Time (s)", "Min Doppler (Hz)", "Max Doppler (Hz)", "Doppler Bandwidth (Hz)", "Peak Signal Strength (dB)", "STD of Signal (dB)"])
    for idx in range(len(t)):
        workbook_features.active.append([float(t[idx]), float(min_doppler_frequency[idx]), float(max_doppler_frequency[idx]),
                                         float(doppler_bandwidth[idx]), float(peak_signal_strength[idx]), float(standard_deviation_signal[idx])])

    #Micro-Doppler peaks min sheet
    micro_doppler_peaks_min_sheet = workbook_features.create_sheet("Micro-Doppler Peaks Min")
    micro_doppler_peaks_min_sheet.append(["Time (s)", "Doppler Frequency (Hz)"])
    for i in range(len(micro_doppler_peaks_min)):
        micro_doppler_peaks_min_sheet.append([float(time_of_micro_doppler_peaks_min[i]), float(micro_doppler_peaks_min[i])])

    #Micro-Doppler peaks max sheet
    micro_doppler_peaks_max_sheet = workbook_features.create_sheet("Micro-Doppler Peaks Max")
    micro_doppler_peaks_max_sheet.append(["Time (s)", "Doppler Frequency (Hz)"])
    for i in range(len(micro_doppler_peaks_max)):
        micro_doppler_peaks_max_sheet.append([float(time_of_micro_doppler_peaks_max[i]), float(micro_doppler_peaks_max[i])])


    #Feature vector sheet
    feature_vector_sheet = workbook_features.create_sheet("Feature Vector")
    feature_titles = ["Min of Min Doppler (Hz)", "Max of Min Doppler (Hz)", "Mean of Min Doppler (Hz)", "STD of Min Doppler (Hz)",
                      "Min of Max Doppler (Hz)", "Max of Max Doppler (Hz)", "Mean of Max Doppler (Hz)", "STD of Max Doppler (Hz)",
                      "Min of Bandwidth (Hz)", "Max of Bandwidth (Hz)", "Mean of Bandwidth (Hz)", "STD of Bandwidth (Hz)",
                      "Min of Peak Signal Strength (dB)", "Max of Peak Signal Strength (dB)", "Mean of Peak Signal Strength (dB)", "STD of Peak Signal Strength (dB)",
                      "Min of STD of Signal (dB)", "Max of STD of Signal (dB)", "Mean of STD of Signal (dB)", "STD of STD of Signal (dB)"]
    for title, value in zip(feature_titles, feature_vector):
        feature_vector_sheet.append([title, value])

    #Cadence frequency
    peak_times = numpy.array(time_of_micro_doppler_peaks_max)

    if len(peak_times) > 1:
        average_time_step = numpy.mean(numpy.diff(peak_times))   #average_time_step = average time step between micro-Doppler peaks
        cadence_frequency = 1 / average_time_step   #cadence frequency = 1/average time step
    else:
        cadence_frequency = 0   #no periodic motion (no micro-Doppler peaks)

    cadence_sheet = workbook_features.create_sheet("Cadence Frequency")
    cadence_sheet.append(["Cadence Frequency (Hz)"])
    cadence_sheet.append([float(cadence_frequency)])
    
    #Full feature vector for activity
    activity_sheet = workbook_features.create_sheet(Activity)
    activity_sheet.append(["Feature", "Value"])
    feature_names = feature_titles + ["Cadence Frequency (Hz)"]
    feature_values = feature_vector + [cadence_frequency]
    for name, value in zip(feature_names, feature_values):
        activity_sheet.append([name, float(value)])

    workbook_features.save(rf"C:\Users\Asnie\Downloads\Dataset_848\6 February 2019 NG Homes Dataset\Features\Features - {FileName}.xlsx")

