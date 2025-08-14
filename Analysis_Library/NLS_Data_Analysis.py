import h5py
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def select_file():
    """
    打开文件选择对话框，选择 HDF5 文件
    返回: 选择的文件路径，如果取消则返回 None
    """
    # create a hidden tkinter root window
    root = tk.Tk()
    root.withdraw()  # hide the main window
    # set the initial directory (if exist)
    initial_dir = r'C:\Users\JeffreyQ\Desktop\UCLA_HW\Physics_Research\Regan_Group\Summer_REU\NLS_HZO'
    if not os.path.exists(initial_dir):
        initial_dir = os.getcwd()
    # open the file selection dialog
    file_path = filedialog.askopenfilename(
        title="选择 HDF5 数据文件",
        initialdir=initial_dir,
        filetypes=[
            ("HDF5 files", "*.hdf5"),
            ("HDF5 files", "*.h5"),
            ("All files", "*.*")
        ]
    )
    # destroy the tkinter root window
    root.destroy()
    return file_path if file_path else None




class NLS_Data():
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.NLS_File = h5py.File(file_path, "r")
        # open and get file data from read only mode
        self.ch1_raw_adc_data = self.NLS_File['children']['0']['raw_data']
        self.ch2_raw_adc_data = self.NLS_File['children']['1']['raw_data']
        self.daq_adc_ref = self.NLS_File['children']['1']['meta']['daq_z_reference'][()]    
        self.daq_adc_increment = self.NLS_File['children']['1']['daq_z_increment'][()]
        self.tia_gain = self.NLS_File['children']['1']['convert_z_increment'][()]
        self.sample_name = self.NLS_File['sample_info']['id'][()]
        self.sample_area = self.NLS_File['sample_info']['DUTs']['0']['area'][()] * (1e-12)
        self.sample_thickness = self.NLS_File['sample_info']['DUTs']['0']['thickness'][()]
        self.sampling_period = self.NLS_File['children']['1']['daq_x_increment'][()]



    def detect_pulse_indices(self, data=None, threshold_factor=0.1, min_pulse_width=50):
        """
        检测数据中两个脉冲的起始和结束索引

        参数:
        - data: 输入数据数组（如果为None，则使用ch1_raw_adc_data）
        - threshold_factor: 阈值因子（默认0.1）
        - min_pulse_width: 最小脉冲宽度（默认50）
        返回:
        - tuple: (pulse1_start, pulse1_end, pulse2_start, pulse2_end) 
                 如果没找到2个脉冲则返回 (None, None, None, None)
        """
        if data is None:
            data = self.ch1_raw_adc_data
        # calculate the baseline
        baseline_start = np.mean(data[:100])
        baseline_end = np.mean(data[-100:])
        baseline = (baseline_start + baseline_end) / 2
        # calculate the data extremum
        data_max = np.max(data)
        data_min = np.min(data)
        # calculate the positive and negative range
        positive_range = data_max - baseline
        negative_range = baseline - data_min
        # determine the pulse type and set the corresponding detection logic
        if positive_range > negative_range:
            # positive pulse detection
            data_range = positive_range
            threshold = baseline + threshold_factor * data_range
            condition = data > threshold
        else:
            # negative pulse detection
            data_range = negative_range
            threshold = baseline - threshold_factor * data_range
            condition = data < threshold
        # find the pulse boundary
        pulse_starts = []
        pulse_ends = []
        in_pulse = False
        pulse_start_idx = 0
        for i in range(1, len(condition)):
            # detect the pulse start
            if not condition[i-1] and condition[i]:
                pulse_start_idx = i
                in_pulse = True
            # detect the pulse end
            elif condition[i-1] and not condition[i] and in_pulse:
                pulse_width = i - pulse_start_idx
                if pulse_width >= min_pulse_width:
                    pulse_starts.append(pulse_start_idx)
                    pulse_ends.append(i-1)
                in_pulse = False
        # process the last pulse
        if in_pulse and len(data) - pulse_start_idx >= min_pulse_width:
            pulse_starts.append(pulse_start_idx)
            pulse_ends.append(len(data)-1)
        # return the result
        if len(pulse_starts) == 2:
            return pulse_starts[0], pulse_ends[0], pulse_starts[1], pulse_ends[1]
        else:
            return None, None, None, None
        


    def polarization_analysis(self):
        """
        极化分析函数 - 计算电场和净极化强度
        参数:
        - input_pulse: 输入脉冲数据（通常是ch1_raw_adc_data）
        - detect_pulse: 检测脉冲数据（通常是ch2_raw_adc_data）
        - daq_adc_ref: ADC参考值（0V对应的ADC值）
        返回:
        - tuple: (E_field, net_polarization) 
                 E_field: 电场强度数组
                 net_polarization: 净极化强度数组
                 如果检测失败则返回 None
        """
        input_pulse = self.ch1_raw_adc_data
        detect_pulse = self.ch2_raw_adc_data
        try:
            # use the detect_pulse_indices method in the class to detect the pulse
            input_start1, input_end1, input_start2, input_end2 = self.detect_pulse_indices()
        except:
            print("No pulse detected")
            return None
        # recalculate the input2_end to ensure the duration of the two pulses is the same
        input_end2 = input_end1 - input_start1 + input_start2
        # extract the detection data of the two pulses and convert to float64 to avoid overflow
        detect_pulse1 = detect_pulse[input_start1:input_end1].astype(np.float64)
        detect_pulse2 = detect_pulse[input_start2:input_end2].astype(np.float64)
        # since the addition and subtraction of detect_pulse exceeds the range of uint16, it needs to be converted to float64
        # calculate the net switch pulse (the difference between the two pulses)
        net_switch_pulse = np.array(detect_pulse1) - np.array(detect_pulse2)
        # convert to the real switch current
        pure_switch_current = (net_switch_pulse * self.daq_adc_increment) * self.tia_gain
        # initialize the net polarization array
        net_polarization = np.zeros(detect_pulse1.size)
        temp_polarization = 0
        # calculate the net polarization strength by integration
        for i in range(net_polarization.size):
            temp_polarization += pure_switch_current[i] * self.sampling_period / self.sample_area
            # the calculation of the I = dQ/dt = d(P*A)/dt, 
            net_polarization[i] = temp_polarization
        # calculate the input voltage and electric field strength
        input_pulse1 = input_pulse[input_start1:input_end1].astype(np.float64)
        input_volt = (np.array(input_pulse1) - np.array(self.daq_adc_ref)) * self.daq_adc_increment
        E_field = input_volt * 1e-8 / self.sample_thickness
        # turn the unit to MV/cm
        return E_field, net_polarization



    def trap_induced_polarization(self):
        """
        通过极化分析结果，找到绝对电场最大的点对应的净极化值。
        返回:
        - tuple: (E_field_at_max_abs, polarization_at_index)
        若分析失败则返回 None
        """
        result = self.polarization_analysis()
        if result is None:
            return None
        E_field, net_polarization = result
        if E_field is None or len(E_field) == 0:
            return None
        max_idx = int(np.argmax(np.abs(E_field)))
        polarization_value = float(net_polarization[max_idx])
        self.trap_induced_P = polarization_value
        try:
            holding_voltage = self.NLS_File['rectangular_template']['voltage_height'][()]
            holding_time = self.NLS_File['rectangular_template']['hold_time'][()]
            return holding_voltage, holding_time , polarization_value
        except:
            return None, None, polarization_value



    def plot_polarization(self):
        E_field, net_polarization = self.polarization_analysis()
        plt.figure(figsize=(12, 6))
        plt.plot(E_field, net_polarization)
        plt.title('Rectangular Pulse Induced Polarization Measured by ND')
        plt.xlabel('Electric Field (MV/cm)')
        plt.ylabel('Polarization (C/m^2)')
        plt.grid(True)
        plt.show()



    def close(self):
        if isinstance(self.NLS_File, h5py.File):
            try:
                self.NLS_File.close()
            except:
                print("Failed to close NLS_File")
        return




