#------------------------------------------------------------------------------------------------------------------
""" EBR file library
    
    This script contains functions for loading and saving RAW and EBR files. 
    
    A RAW data file is used to store EEG recordings without epoching or processing. These files
    contains the following elements:
        Data type - The data type used to store the EEG record (int, double, float, complex, etc.).
        Number of channels - The number of electrode positions.
        Channel names - The names of the electrodes.
        Number of samples - The number of time points of the EEG record.
        Sampling rate - The sampling rate used to record the EEG data.
        Number of comments - The number of comments added by the experimenter.
        Comment list - The list of comments included in the file.
        Number of marks - The number of marks that indicate time events.
        Mark list - The list of marks that indicate the time events. A mark is a pair that indicates
                    the sameple point of the event and the name of the event.
        Data - The array with the record. Rows represent samples and columns represent channels.
    
    On the other hand, an EBR data file is used to store epoched or band-filtered EEG recordings. 
    These files contains the following elements:
        Data type - The data type used to store the EEG record (int, double, float, complex, etc.).
        Number of trials - The number of trials or repetitions.
        Trial names - The names of the trials.
        Number of channels - The number of electrode positions.
        Channel names - The names of the electrodes.
        Number of bands - The number of bands.
        Band names - The names of the bands.
        Number of samples - The number of time points of the EEG record.
        Sampling rate - The sampling rate used to record the EEG data.
        Number of comments - The number of comments added to the file by the experimenter.
        Comment list - The list of comments included in the file.
        Number of marks - The number of marks that indicate time events.
        Mark list - The list of marks that indicate the time events. A mark is a pair that indicates
                    the sameple point of the event and the name of the event.
        Data - The array with the record. The first dimension represents trials, the second dimension 
               represents channels, the third dimension corresponds to bands, and the last dimension 
               represents the time points.

    This library handles the EBR data files in dictionaries, whose elements represent the data 
    described above.

    Author
    ------
    Omar Mendoza Montoya

    Email
    -----
    omendoz@live.com.mx

    Copyright
    ---------
    Copyright (c) 2022 Omar Mendoza Montoya. All rights reserved.
    
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,  
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,  
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
    subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial 
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
#------------------------------------------------------------------------------------------------------------------

import os
import numpy as np

def load_ebr_file(file):
    """ Load EBR file
        
        This function loads an EBR file and returns its content in a dictionary.

        Parameters
        ----------
        file : str
            The name of the file to load.

        Returns
        -------
        dict
           A dictionary with the loaded data.                
    """

    # Check arguments

    if not isinstance(file, str):        
        raise Exception("The argument 'file' must be a string.")

    if not os.path.exists(file):
        raise Exception("The specified path is not valid or does not exit.")        

    # Open file
    data_file = open(file, "rb")

    # Read magic key
    magic = data_file.readline().strip().lower()
    if not magic == b'ebr binary 1.0':
        raise Exception("The specified file is not a binary EBR file.")

    # Read header
    data_type = "double"
    fs = 0
    ns = 0
    nb = 0
    bands = []
    nc = 0
    channels = []
    nt = 0
    trials = []    
    ncomments = 0
    comments = []
    nmarks = 0
    marks = []

    while True:
        line = data_file.readline().strip()
       
        if line.startswith(b'data_type'):
            splited_line = line.split(b'data_type', 1)
            data_type = splited_line[1].strip().decode("utf-8") 

        elif line.startswith(b'sampling_rate'):
            splited_line = line.split(b'sampling_rate', 1)
            fs = float(splited_line[1].strip())

        elif line.startswith(b'samples'):
            splited_line = line.split(b'samples', 1)
            ns = int(splited_line[1].strip())

        elif line.startswith(b'bands'):
            splited_line = line.split(b'bands', 1)
            nb = int(splited_line[1].strip())
            bands = ['']*nb

        elif line.startswith(b'band_'):
            splited_line = line.split(b'band_', 1)
            info = splited_line[1].split(b' ', 1)
            index = int(info[0])-1
            bands[index] = info[1].strip().decode("utf-8")  

        elif line.startswith(b'channels'):
            splited_line = line.split(b'channels', 1)
            nc = int(splited_line[1].strip())
            channels = ['']*nc

        elif line.startswith(b'channel_'):
            splited_line = line.split(b'channel_', 1)
            info = splited_line[1].split(b' ', 1)
            index = int(info[0])-1
            channels[index] = info[1].strip().decode("utf-8")  
            
        elif line.startswith(b'trials'):
            splited_line = line.split(b'trials', 1)
            nt = int(splited_line[1].strip())
            trials = ['']*nt

        elif line.startswith(b'trial_'):
            splited_line = line.split(b'trial_', 1)
            info = splited_line[1].split(b' ', 1)
            index = int(info[0])-1
            trials[index] = info[1].strip().decode("utf-8")  

        elif line.startswith(b'comments'):
            splited_line = line.split(b'comments', 1)
            ncomments = int(splited_line[1].strip())
            comments = ['']*ncomments

        elif line.startswith(b'comment_'):
            splited_line = line.split(b'comment_', 1)
            info = splited_line[1].split(b' ', 1)
            index = int(info[0])-1
            comments[index] = info[1].strip().decode("utf-8")  

        elif line.startswith(b'marks'):
            splited_line = line.split(b'marks', 1)
            nmarks = int(splited_line[1].strip())
            marks = ['']*nmarks

        elif line.startswith(b'mark_'):
            splited_line = line.split(b'mark_', 1)
            info = splited_line[1].split(b' ', 2)
            index = int(info[0])-1
            mark_index = int(info[1])
            marks[index] = (mark_index, info[2].strip().decode("utf-8"))

        elif line.startswith(b'end_header'):
            break

    # Read data
    data_size = nt*nc*nb*ns
     
    if data_type == 'int8' or data_type== 'char':
        raw_data = data_file.read(data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int8))
        data = data.astype('float64')

    elif data_type == 'uint8' or data_type== 'unsigned char':
        raw_data = data_file.read(data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint8))
        data = data.astype('float64')

    elif data_type == 'int16' or data_type== 'short':
        raw_data = data_file.read(2*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int16))
        data = data.astype('float64')

    elif data_type == 'uint16' or data_type== 'unsigned short':
        raw_data = data_file.read(2*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint16))
        data = data.astype('float64')

    elif data_type == 'int32' or data_type== 'int':
        raw_data = data_file.read(4*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int32))
        data = data.astype('float64')

    elif data_type == 'uint32' or data_type== 'unsigned int':
        raw_data = data_file.read(4*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint32))
        data = data.astype('float64')

    elif data_type == 'int64' or data_type== '__int64':
        raw_data = data_file.read(8*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int64))
        data = data.astype('float64')

    elif data_type == 'uint64' or data_type== 'unsigned __int64':
        raw_data = data_file.read(8*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint64))
        data = data.astype('float64')

    elif data_type == 'float':
        raw_data = data_file.read(4*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.float32))
        data = data.astype('float64')

    elif data_type == 'double':
        raw_data = data_file.read(8*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.float64))

    elif data_type == 'complex' or data_type== 'class std::complex<double>':
        raw_data = data_file.read(16*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.cdouble))

    data = data.reshape((nt, nc, nb, ns))

    data_file.close()

    # Build data structure
    ebr_data = dict()
    
    ebr_data["data_type"] = data_type

    ebr_data["sampling_rate"] = fs

    ebr_data["number_of_trials"] = nt
    ebr_data["trials"] = trials

    ebr_data["number_of_channels"] = nc
    ebr_data["channels"] = channels

    ebr_data["number_of_bands"] = nb
    ebr_data["bands"] = bands

    ebr_data["number_of_samples"] = ns
    
    ebr_data["number_of_comments"] = ncomments
    ebr_data["comments"] = comments

    ebr_data["number_of_marks"] = nmarks
    ebr_data["marks"] = marks

    ebr_data["data"] = data

    return ebr_data

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------

def load_ebr_file_to_df(file, end_id: int | None = 101, idle_id: int | None = 0):
    import pandas as pd

    results = load_ebr_file(file)
    sampling_rate = results['sampling_rate']

    df = pd.DataFrame(
        {
            name: column[0]
            for name, column in zip(results['channels'], results['data'][0])
        }
    )
    df.rename(columns={'MARK': 'label'}, inplace=True)
    
    if end_id is not None:
        df['group'] = (df['label'] == end_id).cumsum()
    
    if idle_id is not None:
        df['label'] = df['label'].replace(idle_id, pd.NA).ffill()
        df.dropna(inplace=True)

    return df, sampling_rate