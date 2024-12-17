# PIMProcessor and Data Handling Documentation

This document serves as a detailed guide to understand and utilize the provided code for IoT battery data extraction, processing, and modeling using PIMProcessor. 

---

## **Overview**

The code includes multiple functions and a class `PIMProcessor` for:
1. Fetching IoT battery data.
2. Processing the data into usable formats.
3. Generating final tables for SoC and SoH comparison.
4. Running PIM models using provided directory paths and configurations.

---

## **Authentication**

The code authenticates via the `.ligrc` file stored in the home directory. Ensure the following:
- The file `.ligrc` contains `auth_key==<your_auth_key>`.
- Authentication enables access to APIs for retrieving and processing IoT data.

---

## **Function Descriptions**

### `get_imei(IMEI)`
Fetches the IMEI from the dashboard.

#### Parameters:
- `IMEI`: IMEI code of the battery.

#### Returns:
- The IMEI string.

### `get_dates(start_date1, end_date1)`
Generates a list of dates between `start_date1` and `end_date1`.

#### Parameters:
- `start_date1`: Starting date.
- `end_date1`: Ending date.

#### Returns:
- List of dates in `YYYY-MM-DD` format.

### `get_extdata(IMEI, start_time, end_time, filter_data=False)`
Fetches external IoT data.

#### Parameters:
- `IMEI`: IMEI code of the battery.
- `start_time`: Start timestamp.
- `end_time`: End timestamp.
- `filter_data`: (Optional) Boolean to filter data by timestamps.

#### Returns:
- DataFrame containing IoT data.

### `get_pimdata()`
Processes IoT dashboard data for Local PIM testing.

#### Parameters:
- `IMEI`, `start_time`, `end_time`, `filter_data`, `serial_no`, `interpolation`, `period`, `nas`.

#### Returns:
- Processed DataFrame with interpolated voltage and current data.

### `get_fwdata(fWVersion, battery_prefix)`
Fetches packs with specific firmware versions.

#### Parameters:
- `fWVersion`: Firmware version string.
- `battery_prefix`: Battery prefix string.

#### Returns:
- DataFrame with firmware-specific data.

### `authenticate()`
Authenticates and returns a session ID.

#### Returns:
- Session ID string.

### `read_file_in_memory(sid, imei, year, month, day, file_date)`
Reads a `.parquet` file from the NAS storage into memory.

#### Parameters:
- `sid`: Session ID.
- `imei`, `year`, `month`, `day`, `file_date`: File location parameters.

#### Returns:
- DataFrame with file data.

### `adjust_end_date(end_date)`
Adjusts the end date if it exceeds today.

#### Parameters:
- `end_date`: Input end date string.

#### Returns:
- Adjusted end date string.

### `filter_datas(df, start_time, end_time)`
Filters a DataFrame between `start_time` and `end_time`.

#### Parameters:
- `df`: DataFrame to filter.
- `start_time`, `end_time`: Timestamp strings.

#### Returns:
- Filtered DataFrame.

### `get_data(imei, start_time, end_time, serial_no=False, filter_data=False, skip=False)`
Fetches and combines battery data from NAS storage.

#### Parameters:
- `imei`, `start_time`, `end_time`.
- `serial_no`, `filter_data`, `skip`.

#### Returns:
- Combined DataFrame.

### `pim_make(directory_path)`
Runs PIM models by invoking the `make` command.

#### Parameters:
- `directory_path`: Path to the directory containing the code.

---

## **Class: PIMProcessor**

Processes PIM models and generates tables for SoC and SoH comparisons.

### **Methods**

#### `__init__(directory_path)`
Initializes the PIMProcessor with a directory path.

#### `fetch_and_process_data(serial_numbers, start_date, end_date, checker="oh", nas=False)`
Processes IoT data for multiple serial numbers.

#### Parameters:
- `serial_numbers`: List of serial numbers.
- `start_date`, `end_date`: Date range for processing.
- `checker`: Specifies SoC (`'oc'`) or SoH (`'oh'`).
- `nas`: Boolean for data source (NAS or external).

#### `generate_final_table(checker)`
Generates a final table for SoC or SoH.

#### `plot_soh(checker)`
Plots SoH comparisons for multiple packs.

---

## **Example Usages**

### 1. **Fetch IMEI Data**
```python
imei = get_imei("MD0AIOALAA00638")
```

### 2. **Get IoT Data**
```python
odf = get_extdata("MD0AIOALAA00638", "2024-06-27", "2024-06-28")
```

### 3. **Run PIM Make Command**
```python
pim_make("/path/to/directory")
```

### 4. **Process PIM Data**
```python
processor = PIMProcessor("/path/to/directory")
processor.fetch_and_process_data(
    serial_numbers=["MD0AIOALAA00638"],
    start_date="2024-10-25",
    end_date="2024-10-26",
    checker="oh",
    nas=False
)
processor.generate_final_table(checker="oh")
processor.plot_soh(checker="oh")
```

---

## **Dependencies**

- Python 3.6+
- Libraries: `json`, `io`, `concurrent.futures`, `os`, `requests`, `subprocess`, `pandas`, `numpy`, `plotly`, `tqdm`.

---

## **Notes**
- Ensure correct configurations in `.ligrc` for authentication.
- Use proper directory structure for PIM modeling.
- Handle exceptions for missing or invalid files while fetching NAS data.

---

### For queries, contact the author or check the source repository for updates.
