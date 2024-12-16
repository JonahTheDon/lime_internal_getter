import json
import io
import concurrent.futures
import os
import requests
import subprocess
import pandas as pd
from datetime import timedelta, date
import numpy as np
import plotly.graph_objects as go
from tqdm.notebook import tqdm

home = os.path.expanduser("~")
filename = ".ligrc"
path = os.path.join(home, filename)
try:
    with open(path, "r") as file:
        content = file.read()
except FileNotFoundError as e:
    print(e)
    print("Please make sure that auth_key==<auth_key> is stored in above file.")
    print("Required for BAP login!")
    exit()

auth_key = content.strip().split("==")[1]

# Login API endpoint
login_api = "http://10.10.4.40:5000/webapi/auth.cgi"
login_params = {
    "api": "SYNO.API.Auth",
    "version": "3",
    "method": "login",
    "account": "harisankar",
    "passwd": "8P1GTw#f",
    "session": "FileStation",
    "format": "sid",
}


def get_imei(IMEI):
    """
    Function to get_imei from dashboard
    ```
    import correction_monitor as cm
    eg: imei=get_imei("MD0AIOALAA00638")
    ```
    """
    headers = headers = {
        "Authorization": f"{auth_key}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"https://api-stage.lime.ai/lime/liveBatteryDashboard/{IMEI}",
        headers=headers,
    )
    try:
        return json.loads(response.content)["result"][0]["imei"]
    except Exception as e:
        raise ValueError("get_imei error: ", e)


def get_dates(start_date1, end_date1):
    """
    Returns a list of dates between the start and end dates, inclusive.

    Parameters:
    - start_date1: starting date
    - end_date1: ending date

    Returns:
    - List of date strings in the format 'YYYY-MM-DD'
    """
    start_day = int(start_date1.split("-")[2].strip())
    start_month = int(start_date1.split("-")[1].strip())
    start_year = int(start_date1.split("-")[0].strip())
    end_day = int(end_date1.split("-")[2].strip())
    end_month = int(end_date1.split("-")[1].strip())
    end_year = int(end_date1.split("-")[0].strip())
    # Convert start and end dates to datetime objects
    start_date = date((start_year), (start_month), (start_day))
    end_date = date((end_year), (end_month), (end_day))

    # List to hold date strings
    date_list = []

    # Iterate through each day in the range
    current_date = start_date
    while current_date <= end_date:
        # Format date as 'YYYY-MM-DD'
        date_str = current_date.strftime("%Y-%m-%d")
        date_list.append(date_str)

        # Move to the next day
        current_date += timedelta(days=1)

    return date_list


def get_extdata(IMEI, start_time, end_time, filter_data=False):
    """
    Function to import data from dashboard
    ```
    import lime_internal_getter as ig
    odf = ig.get_extdata(*("MD0AIOALAA00638", "2024-06-27", "2024-06-28"))
    ```
    """
    start_date = start_time.split(" ")[0]
    start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date = end_time.split(" ")[0]
    start_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    headers = headers = {
        "Authorization": f"{auth_key}",
        "Content-Type": "application/json",
    }
    payload = json.dumps(
        {"imei": IMEI, "startDate": start_date, "endDate": end_date}
    )
    response = requests.post(
        "https://api-stage.lime.ai/lime/iotData", headers=headers, data=payload
    )
    try:
        df = pd.DataFrame(json.loads(response.content)["result"])
        if filter_data is True:
            df["timeStamp"] = pd.to_datetime(df["timeStamp"])
            df = (
                df[
                    (df["timeStamp"] >= start_time)
                    & (df["timeStamp"] <= end_time)
                ]
            ).reset_index(drop=True)
        return df
    except Exception:
        raise ValueError("Cannot Form Dataframe")


def get_pimdata(
    IMEI,
    start_time,
    end_time,
    filter_data=False,
    serial_no=False,
    interpolation=True,
    period=0.1,
    nas=True,
):
    """
    Function to get data from IoT dashboard for Local PIM testing
    ```
    import lime_internal_getter as ig
    odf = ig.get_extdata(*("MD0AIOALAA00638", '2024-10-25 14:30', '2024-10-26 02:17'))
    ```
    """
    if nas:
        df = get_data(
            IMEI,
            start_time,
            end_time,
            filter_data=filter_data,
            serial_no=serial_no,
        )
        df["timeStamp"] = df["date"] + " " + df["time"]
    else:
        df = get_extdata(IMEI, start_time, end_time, filter_data=filter_data)
    df["Time diff"] = (
        (pd.to_datetime(df["timeStamp"]).astype("int64") / 10**9)
        .diff()
        .fillna(0)
    )
    df["Cumulative Time"] = df["Time diff"].cumsum().fillna(0)
    col_names = [col for col in df.columns if "Volt" in col and "cell" in col]
    if interpolation:
        time = np.arange(0, df["Cumulative Time"].iloc[-1], period)
        data = [
            pd.Series(time),
            pd.Series(np.interp(time, df["Cumulative Time"], df["batCurrent"])),
        ]
        for col in col_names:
            if not (df[col].eq(0).all()):
                data.append(
                    pd.Series(
                        np.interp(time, df["Cumulative Time"], df[col] / 1000)
                    )
                )
    else:
        data = [df["Cumulative Time"], df["batCurrent"]]
        for col in col_names:
            if not (df[col].eq(0).all()):
                data.append(df[col] / 1000)
    return pd.DataFrame(data).T


def get_fwdata(fWVersion="8183D", battery_prefix="MH"):
    """
    Use for getting list of packs with different firmware versions
    import lime_internal_getter as ig
    eg: df= ig.get_fwdata(fWVersion="8183D",battery_prefix="MH",date="2024-11-15")
    """
    headers = headers = {
        "Authorization": f"{auth_key}",
        "Content-Type": "application/json",
    }
    payload = json.dumps(
        {
            "batteryPrefix": battery_prefix,
            "filterParams": [
                {
                    "parameterName": "fwVersion",
                    "downloadType": "Exact",
                    "value": fWVersion,
                }
            ],
            "selectedParams": [
                "socPercent",
                "maxCellSocE",
                "cellSoc3E",
                "minCellSocE",
                "maxCellSohE",
                "minCellSohE",
                "trueSoc",
                "cellSoh3E",
                "socPercentE",
            ],
            "type": "Download",
        }
    )
    response = requests.post(
        "https://api-stage.lime.ai/lime/liveparams/filterDispatchCheckDownloadData",
        headers=headers,
        data=payload,
    )
    return pd.DataFrame(json.loads(response._content)["result"])


def authenticate():
    # Login parameters
    login_params2 = {
        "api": "SYNO.API.Auth",
        "version": "3",
        "method": "login",
        "account": "harisankar",
        "passwd": "8P1GTw#f",
        "session": "FileStation",
        "format": "sid",
    }
    response = requests.get(login_api, params=login_params2)
    data = response.json()
    if data.get("success", False):
        # print("Login successful.")
        return data["data"]["sid"]
    else:
        print("Login failed:", data)
        return None


file_api = "http://10.10.4.40:5000/webapi/entry.cgi"


# Function to read a file via API and process it in memory
def read_file_in_memory(sid, imei, year, month, day, file_date):
    file_path = f"/lime-datalake-prod/lime_bap_parquet/{imei}/{year}/{month}/{day}/{file_date}.parquet"
    file_params = {
        "api": "SYNO.FileStation.Download",
        "version": "2",
        "method": "download",
        "path": file_path,
        "_sid": sid,
    }

    response = requests.get(file_api, params=file_params, stream=True)
    if response.ok:
        try:
            # Read the file into a DataFrame
            file_stream = io.BytesIO(response.content)
            df = pd.read_parquet(file_stream)
            if not isinstance(df, pd.DataFrame):
                raise ValueError("df returned: ", df)
            return df
        except Exception as e:
            raise ValueError(e)
    else:
        raise ValueError(
            "Response: ", response.ok
        )  # Skip printing errors for missing files


def adjust_end_date(end_date: str) -> str:
    """
    Adjusts the end_date if it is greater than or equal to today's date.
    Sets it to yesterday's date in '%d-%m-%Y' format.

    Parameters:
    end_date (str): The end date in '%d-%m-%Y' format.

    Returns:
    str: The adjusted end date in '%d-%m-%Y' format.
    """
    # Parse the input end_date
    end_date_parsed = pd.to_datetime(end_date, format="%Y-%m-%d")
    today = pd.Timestamp.today().normalize()

    # Check if the end_date is >= today
    if end_date_parsed >= today:
        # Set to yesterday
        end_date_parsed = today - pd.Timedelta(days=1)

    # Return the adjusted date in '%d-%m-%Y' format
    return end_date_parsed.strftime("%Y-%m-%d")


def filter_datas(df, start_time, end_time):
    """
    This filters data from start time to end time
    ## Example Usage ##
    df= filter_data(df, '2024-10-25 14:30', '2024-10-26 02:17')

    """
    timestamp = pd.to_datetime(df["date"] + " " + df["time"])
    df = (df[(timestamp >= start_time) & (timestamp <= end_time)]).reset_index(
        drop=True
    )


def get_data(
    imei, start_time, end_time, serial_no=False, filter_data=False, skip=False
):
    """
    Use for getting battery data from NAS storage eg:
    df = get_data("MD0AIOALAA00638", '2024-10-25 14:30', '2024-10-26 02:17', filter_data=True, serial_no=True)
    """
    start_date = start_time.split(" ")[0]
    end_date = end_time.split(" ")[0]

    if serial_no:
        try:
            imei = get_imei(imei)
        except Exception:
            raise ValueError("Can't get IMEI")

    end_date = adjust_end_date(end_date)

    sid = authenticate()
    if not sid:
        print("Exiting due to authentication failure.")
        raise ValueError("Authentication Failure")

    # Get dates and ensure they are sorted
    dates = pd.to_datetime(get_dates(start_date, end_date))
    dates = dates.sort_values()

    # Precompute date components to minimize repeated operations
    date_components = [
        (
            d.strftime("%Y"),
            d.strftime("%m").lstrip("0"),
            d.strftime("%d").lstrip("0"),
            d.strftime("%Y%m%d"),
        )
        for d in dates
    ]

    def read_file_wrapper(args):
        """Wrapper function for reading files in parallel."""
        year, month_lstrip, day_lstrip, file_date = args
        try:
            return (
                file_date,
                read_file_in_memory(
                    sid, imei, year, month_lstrip, day_lstrip, file_date
                ),
            )
        except Exception:
            if skip:
                return None
            else:
                raise ValueError(f"Issue getting data for the date {file_date}")

    # Use ThreadPoolExecutor for parallel file reading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(read_file_wrapper, components)
            for components in date_components
        ]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                if future.result():
                    results.append(future.result())
            except Exception as e:
                raise e

    # Sort results by file_date to maintain order
    results.sort(key=lambda x: x[0])

    # Extract data frames in the correct order
    data_frames = [result[1] for result in results]

    # Concatenate all data frames in order
    df = pd.concat(data_frames, ignore_index=True)

    if filter_data:
        filter_datas(df, start_time, end_time)

    return df.reset_index(drop=True)


class BatterySOXProcessor:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.fdf = None
        self.final_table = None

    def fetch_and_process_data(
        self, serial_numbers, start_date, end_date, checker="oh"
    ):
        k = 0
        original_directory = os.getcwd()

        for ser in tqdm(serial_numbers, desc="Processing Serial Numbers"):
            params = (ser, start_date, end_date)

            try:
                # Fetch IoT data and save as CSV
                get_pimdata(
                    params[0], params[1], params[2], nas=False, serial_no=True
                ).to_csv(
                    f"{self.directory_path}/input_data/drive_cycle_iot_data.csv",
                    index=False,
                    header=None,
                )
            except Exception as e:
                print(f"Error fetching IoT data for {ser}: {e}")
                continue

            try:
                # Change to directory and run make command
                os.chdir(self.directory_path)
                subprocess.run(
                    ["make"], check=True, text=True, capture_output=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error running make for {ser}: {e.stderr}")
                continue
            finally:
                # Change back to the original directory
                os.chdir(original_directory)

            try:
                df = get_extdata(params[0], params[1], params[2])
            except Exception as e:
                print(f"Error fetching extended data for {ser}: {e}")
                continue

            df["Time diff"] = (
                (pd.to_datetime(df["timeStamp"]).astype("int64") / 10**9)
                .diff()
                .fillna(0)
            )

            df["Cumulative Time"] = df["Time diff"].cumsum().fillna(0)
            time_data = pd.to_datetime(df["timeStamp"]).astype("int64") / 10**9

            odf = pd.read_csv(
                f"{self.directory_path}/output_data/final_s{checker}_iot_data.csv",
                header=None,
            )

            maxx = odf.iloc[:, 1:].max(axis=1)
            minn = odf.iloc[:, 1:].min(axis=1)

            df[f"PIM_S{checker}"] = np.interp(
                df["Cumulative Time"], odf[0], odf[3]
            )
            df[f"PIM_maxS{checker}"] = np.interp(
                df["Cumulative Time"], odf[0], maxx
            )
            df[f"PIM_minS{checker}"] = np.interp(
                df["Cumulative Time"], odf[0], minn
            )

            df["Cumulative Time"] = pd.to_datetime(time_data, unit="s")
            df["Serial_no"] = ser

            if k > 0:
                self.fdf = pd.concat([self.fdf, df])
            else:
                self.fdf = df
                k += 1

    def generate_final_table(self, checker):
        self.final_table = None
        k = 0

        for ser in tqdm(
            self.fdf["Serial_no"].unique(), desc="Generating Final Table"
        ):
            table = self.fdf[self.fdf["Serial_no"] == ser][:-2:-1].copy()

            for col in [col for col in table.columns if "PIM_" in col]:
                table[col] *= 100

            soh_cols = [
                col
                for col in table.columns
                if checker in col and "Raw" not in col and "reserve" not in col
            ]

            if k == 0:
                self.final_table = table[soh_cols]
                k += 1
            else:
                self.final_table = pd.concat(
                    [self.final_table, table[soh_cols]]
                )

    def plot_soh(self, checker):
        soh_cols = [
            col
            for col in self.fdf.columns
            if checker in col and "Raw" not in col and "reserve" not in col
        ]

        for ser in self.fdf["Serial_no"].unique():
            table = self.fdf[self.fdf["Serial_no"] == ser].copy()

            for col in [col for col in table.columns if "PIM_" in col]:
                table[col] *= 100

            table_data = table[soh_cols]

            fig = go.Figure(layout=dict(title=f"SOH Comparison for {ser}"))

            for tcol in table_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=table["Cumulative Time"],
                        y=table_data[tcol],
                        name=tcol,
                    )
                )

            fig.show(renderer="browser")
