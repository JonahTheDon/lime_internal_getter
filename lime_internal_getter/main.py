import requests
import os
import json
import pandas as pd
import io
from datetime import timedelta, date
import numpy as np

home = os.path.expanduser("~")
filename = ".ligrcw"
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
    Function to import data from dashboard
    ```
    import correction_monitor as cm
    odf = cm.get_data(*("MD0AIOALAA00638", "2024-06-27", "2024-06-28"))
    ```
    """
    headers = headers = {
        "Authorization": f"{auth_key}",
        "Content-Type": "application/json",
    }
    formatted_date = pd.to_datetime("today").strftime("%Y-%m-%d")
    start_d, end_d = formatted_date, formatted_date
    payload = json.dumps({"imei": IMEI, "startDate": start_d, "endDate": end_d})
    key = False
    day_count = 0
    while key is False:
        response = requests.post(
            "https://api-stage.lime.ai/lime/iotData",
            headers=headers,
            data=payload,
        )
        try:
            result = json.loads(response.content)["result"][0]["imei"]
            key = True
            return result
        except Exception as e:
            print("Error: ", e)
            formatted_date = pd.to_datetime(
                formatted_date, format="%Y-%m-%d"
            ) - pd.Timedelta(days=1)
            day_count += 1
            if day_count > 420:
                raise ValueError("Can't get IMEI")


def get_dates(start_year, start_month, start_day, end_year, end_month, end_day):
    """
    Returns a list of dates between the start and end dates, inclusive.

    Parameters:
    - start_year: The year of the start date
    - start_month: The month of the start date
    - start_day: The day of the start date
    - end_year: The year of the end date
    - end_month: The month of the end date
    =9i07u890
        - end_day: The day of the end date

    Returns:
    - List of date strings in the format 'YYYY-MM-DD'
    """
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


def get_pimdata(IMEI, start_time, end_time, filter_data=False, serial_no=False):
    """
    Function to get data from IoT dashboard for Local PIM testing
    ```
    import lime_internal_getter as ig
    odf = ig.get_extdata(*("MD0AIOALAA00638", '2024-10-25 14:30', '2024-10-26 02:17'))
    ```
    """
    df = get_data(
        IMEI, start_time, end_time, filter_data=filter_data, serial_no=serial_no
    )
    df["Time diff"] = (
        (pd.to_datetime(df["timeStamp"]).astype("int64") / 10**9)
        .diff()
        .fillna(0)
    )
    df["Cumulative Time"] = df["Time diff"].cumsum().fillna(0)
    col_names = [col for col in df.columns if "Volt" in col and "cell" in col]
    time = np.arange(0, df["Cumulative Time"].iloc[-1], 0.1)
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


def filter_data(df, start_time, end_time):
    """
    This filters data from start time to end time
    ## Example Usage ##
    df= filter_data(df, '2024-10-25 14:30', '2024-10-26 02:17')

    """
    timestamp = pd.to_datetime(df["date"] + " " + df["time"])
    df = (df[(timestamp >= start_time) & (timestamp <= end_time)]).reset_index(
        drop=True
    )


def get_data(imei, start_time, end_time, serial_no=False, filter_data=False):
    """
    Use for getting battery data from NAS storage eg:
    df= get_data("MD0AIOALAA00638", '2024-10-25 14:30', '2024-10-26 02:17',filter_data=True,serial number =True)
    """
    start_date = start_time.split(" ")[0]
    end_date = end_time.split(" ")[0]
    if serial_no is True:
        try:
            imei = get_imei(imei)
        except Exception as e:
            print("Error: ", e)
            raise ValueError("Can't get IMEI")
    end_date = adjust_end_date(end_date)
    sid = authenticate()
    if not sid:
        print("Exiting due to authentication failure.")
        raise ValueError("Authentication Failure")
    start_day = int(start_date.split("-")[2].strip())
    start_month = int(start_date.split("-")[1].strip())
    start_year = int(start_date.split("-")[0].strip())
    end_day = int(end_date.split("-")[2].strip())
    end_month = int(end_date.split("-")[1].strip())
    end_year = int(end_date.split("-")[0].strip())
    dates = pd.DataFrame(
        pd.to_datetime(
            get_dates(
                start_year, start_month, start_day, end_year, end_month, end_day
            )
        ),
        columns=["timestamp"],
    )
    # Extract date, month, and year as separate strings
    dates["date"] = dates["timestamp"].dt.strftime("%d")
    dates["month"] = dates["timestamp"].dt.strftime("%m")
    dates["year"] = dates["timestamp"].dt.strftime("%Y")
    kdf = 0
    for _date in range(dates.shape[0]):
        year, month, day = (
            dates["year"].iloc[_date],
            dates["month"].iloc[_date],
            dates["date"].iloc[_date],
        )
        file_date = year + month + day
        month_lstrip = month.lstrip("0")
        day_lstrip = day.lstrip("0")
        try:
            idf = read_file_in_memory(
                sid, imei, year, month_lstrip, day_lstrip, file_date
            )
            if kdf == 0:
                df = idf
            kdf += 1
        except Exception:
            continue
            raise ValueError(
                f"Issue getting data for the {dates['timestamp'][_date]}"
            )
        try:
            if kdf > 0 and df is not None:
                df = pd.concat([df, idf])
        except Exception as e:
            raise ValueError("Error: ", e)
            # continue
    if filter_data is True:
        filter_data(df, start_time, end_time)
    return df.reset_index(drop=True)
