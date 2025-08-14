import h5py
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import NLS_Data_Analysis
# remember to first convert the .ipynb file to .py file, then u can directly import the .py file
# and do need to acquire the working directory
# from . import NLS_Data_Analysis
# do not include initialize a python package here.



class storage_data():
    def __init__(self, selected_path: str, storage_file_name: str):  
        self.selected_path = selected_path
        self.storage_file_name = storage_file_name
        self.data_file_count = None
        self.data_path_list = self.count_data_file()
        # first initialize the data_file_count
        # otherwise the count always none when creating storage file
        self.storage_file_path = self.create_storage_file()   



    def count_data_file(self):
        data_path_list = []
        hdf5_patterns = ['*.hdf5', '*.h5', '*.hdf']
        for pattern in hdf5_patterns:
            for p in Path(self.selected_path).glob(pattern):  # only serach for the current directory, not including the sub-folders
                if p.is_file():
                    data_path_list.append(p)
        print(f"一共有{len(data_path_list)}个文件")
        self.data_file_count = len(data_path_list)
        if not data_path_list:
            print("没有找到数据文件")
            return []
        return data_path_list



    def create_storage_file(self):
        selected_folder = Path(self.selected_path)
        # create a sub-folder for storing the analysis results
        sub_folder_path = self.selected_path + "/" + selected_folder.name + '_NLS_Trend'# the sub-folder path
        try:
            Path(sub_folder_path).mkdir(exist_ok=True)    # create the sub-folder (if not exist)
        except:
            print(f"文件夹 {sub_folder_path} 已存在")
            return   
        print(f"创建存储文件夹: {sub_folder_path}")
        try:
            storage_path = str(Path(sub_folder_path) / f"{self.storage_file_name}.hdf5")
            NLS_trend = h5py.File(storage_path, 'w')
            print(f"it's writing mode")
        except:
            NLS_trend = h5py.File(storage_path, 'a')
            print(f"it's appending mode")
            return NLS_trend       
        # create the data group
        data_group = NLS_trend.create_group('Data')
        # create the dataset according to the data_file_count
        dataset_2d = data_group.create_dataset(
            'NLS_Trend',
            shape=(self.data_file_count, 3),
            dtype=np.float64,
            chunks=True,            # enable the chunk storage
        )
        # add the dataset attribute (metadata)
        dataset_2d.attrs['description'] = '2D NLS Trend Data - Rows: Holding Voltages, Columns: Holding Time vs. Polarization'
        dataset_2d.attrs['units'] = 'Voltage: V, Holding Time: s, Polarization: C/m²'
        print(f"已创建HDF5文件: {storage_path}")    
        return storage_path



    def store_nls_data(self):
        first_sample = NLS_Data_Analysis.NLS_Data(str(self.data_path_list[0]))
        # use the with statement to safely create the sample_info
        with h5py.File(self.storage_file_path, 'a') as sample_file:
            # create the sample_info group (if not exist)
            if 'sample_info' not in sample_file:
                sample_info = sample_file.create_group('sample_info')
                # the correct create_dataset syntax - need to specify the data parameter
                string_dtype = h5py.string_dtype(encoding='utf-8', length=100)
                # convert tuple sample_name to string
                sample_info.create_dataset('sample_name', data=str(first_sample.sample_name), dtype=string_dtype)           
                sample_info.create_dataset('area', data=first_sample.sample_area, dtype=np.float64)
                sample_info.create_dataset('area_unit', data='m^2', dtype=string_dtype)
                sample_info.create_dataset('thickness', data=first_sample.sample_thickness, dtype=np.float64)
                sample_info.create_dataset('thickness_unit', data='m', dtype=string_dtype)
                # use the first sample as the represent of all the trials conducted on that sample   
        first_sample.close()
        for file_index in range(self.data_file_count):
            # iterate through all the hdf5 file paths
            try:
                print(f"\n正在处理文件: {self.data_path_list[file_index].name}")  
                temp_nls_data = NLS_Data_Analysis.NLS_Data(str(self.data_path_list[file_index]))
                temp_voltage, temp_hold_time, temp_polarization = temp_nls_data.trap_induced_polarization()
                # automatically group the data by the voltage
                storage_file = h5py.File(self.storage_file_path, 'a')
                # so far the code is working
                NLS_data = storage_file['Data']['NLS_Trend']
                print(temp_voltage, temp_hold_time, temp_polarization)
                NLS_data[file_index, 0] = float(temp_voltage)
                # this is the correct hdf5 assignment structure! not like [file_index, x]
                NLS_data[file_index, 1] = float(temp_hold_time)
                NLS_data[file_index, 2] = float(temp_polarization)
                # need to convert to float64 type to store
                print(NLS_data[file_index])
                # give up the grouping, directly store the data in the format of voltage, time, polarization
                storage_file.flush()  # refresh the buffer
            except Exception as e:
                print(f"处理文件 {self.data_path_list[file_index].name} 时出错: {e}")
                continue  
        storage_file.close()  # close the main file
        print(f"已将数据存储到{self.storage_file_path}")
        return



def select_folder():
    """
    选择文件夹并返回一个生成器，迭代文件夹中的所有HDF5文件
    每次迭代返回一个NLS_Data对象

    返回:
    - generator: 生成NLS_Data对象的生成器
    """
    # create the hidden tkinter root window
    root = tk.Tk()
    root.withdraw()
    # set the initial directory
    initial_dir = r'C:\Users\JeffreyQ\Desktop\UCLA_HW\Physics_Research\Regan_Group\Summer_REU\NLS_HZO'
    if not os.path.exists(initial_dir):
        initial_dir = os.getcwd()
    # open the folder selection dialog
    selected_path = filedialog.askdirectory(
        title="选择包含HDF5文件的文件夹",
        initialdir=initial_dir
    )
    # destroy the tkinter root window
    root.destroy()    
    if not selected_path:
        print("未选择文件夹")
        return
    return str(selected_path), Path(selected_path).name










