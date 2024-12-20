# lime_internal_getter- A Data Handling Library for Lime.ai Documentation

## Overview
This document provides a description of the functions defined in the `lime_internal_getter` package. The functions facilitate various operations including data fetching, IoT dashboard integration, data filtering, and process execution.

---

## Functions

### 1. **get_imei**
**Description:** Retrieves the IMEI of a battery pack from the Lime.ai dashboard.

**Parameters:**
- `IMEI` (str): The IMEI string of the device.

**Returns:**
- IMEI (str): Retrieved IMEI of the battery pack.

**Example Usage:**
```python
imei = get_imei("MD0AIOALAA00638")
```

---

### 2. **get_pimdata**
**Description:** Retrieves and processes IoT dashboard data for Local PIM testing.(Only for developers of PIM)

**Parameters:**
- `IMEI` (str): IMEI of the device.
- `start_time` (str): Start time in `YYYY-MM-DD HH:MM` format.
- `end_time` (str): End time in `YYYY-MM-DD HH:MM` format.
- `filter_data` (bool): Whether to filter data. Default is `False`.
- `serial_no` (bool): Whether the input is a serial number. Default is `False`.
- `interpolation` (bool): Whether to interpolate data. Default is `True`.
- `period` (float): Interpolation period in seconds. Default is `0.1`.
- `nas` (bool): Use NAS storage if `True`. Default is `True`.

**Returns:**
- DataFrame: A pandas DataFrame containing processed data.

**Example Usage:**
```python
pim_data = get_pimdata("MD0AIOALAA00638", "2024-10-25 14:30", "2024-10-26 02:17")
```

---

### 3. **get_fwdata**
**Description:** Fetches a list of packs with specific firmware versions.

**Parameters:**
- `fWVersion` (str): Firmware version to filter.
- `battery_prefix` (str): Battery prefix for filtering.

**Returns:**
- DataFrame: A pandas DataFrame containing the filtered data.

**Example Usage:**
```python

fw_data = get_fwdata(fWVersion="8183D", battery_prefix="MH")
```
---

### 4. **get_data**
**Description:** Fetches battery data from NAS storage or IoT dashboard.

**Parameters:**
- `imei` (str): IMEI or serial number.
- `start_time` (str): Start time in `YYYY-MM-DD HH:MM` format.
- `end_time` (str): End time in `YYYY-MM-DD HH:MM` format.
- `filter_data` (bool): Whether to filter data. Default is `False`.
- `skip` (bool): Skip missing files if `True`. Default is `False`.
- `nas` (bool): Use NAS storage if `True`. Default is `True`.

**Returns:**
- DataFrame: A pandas DataFrame containing fetched data.

**Example Usage:**
```python
battery_data = get_data("MD0AIOALAA00638", "2024-10-25 14:30", "2024-10-26 02:17")
```

---

### 5. **pim_make**
**Description:** Executes the PIM model after setting configurations in C code.(Only for developers of PIM)

**Parameters:**
- `directory_path` (str): Path to the directory containing PIM configurations.
- `model` (int): Model type. Default is `4`.
- `filename` (str): File extension for the output. Default is `"_iot_data.csv"`.

**Example Usage:**
```python
pim_make("/path/to/directory", model=4, filename="_iot_data.csv")
```

---

### 6. **PIMProcessor** (Class)
**Description:** Class for processing PIM models and generating reports. (Not required for non-developers of PIM Model)

**Methods:**
- `__init__(directory_path, model)`: Initializes the processor.
- `fetch_and_process_data(...)`: Processes data for a list of serial numbers.
- `generate_final_table(checker,save_csv)`: Generates the final data table.
- `plot_soh(checker)`: Plots SOH comparison for packs.

**Example Usage:**
```python
processor = PIMProcessor("/path/to/directory", model=4)
processor.fetch_and_process_data([...], "2024-06-27", "2024-06-28")
