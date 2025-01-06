import numpy as np
import pandas as pd


class KalmanFilter:
    def __init__(self, model):
        self.model = model
        if self.model == 9:  # GangFeng 100Ah LFP parameters
            # State transition matrix 1x1
            self.Q = np.array([0.1])
            # Observation matrix 1x1
            self.H = np.array([1])
            # Process noise covariance 1x1
            self.R = np.array([4.71290239e3])
            # SOC(x3) vs OCV(y3) curve
            self.x3 = np.array(
                [
                    0.0,
                    0.03,
                    0.05,
                    0.1,
                    0.15,
                    0.2,
                    0.25,
                    0.3,
                    0.35,
                    0.4,
                    0.45,
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                    0.98,
                    1.0,
                ]
            )
            self.y3 = np.array(
                [
                    2.9076,
                    3.1493,
                    3.20125,
                    3.21665,
                    3.2408,
                    3.26405,
                    3.28,
                    3.29235,
                    3.297,
                    3.29795,
                    3.29905,
                    3.3002,
                    3.3019,
                    3.3051,
                    3.33085,
                    3.3362,
                    3.3364,
                    3.33665,
                    3.33665,
                    3.33625,
                    3.33575,
                    3.33505,
                    3.4006,
                ]
            )
            self.capacity = 100  # Ah
            self.num_cells_parallel = 1
            self.num_cells_series = 16
            self.soh = [1.0 for i in range(len(self.num_cells_series))]

    def update_kalman_filter(
        self, measurement, initial_state, initial_covariance, F, H, Q, R
    ):
        """
        Updates the state and covariance of a Kalman filter given a new measurement.

        Args:
            measurement (float): The new measurement to update the filter with.
            self.initial_state (float): The initial state estimate.
            initial_covariance (float): The initial covariance estimate.
            F (float): The state transition matrix.
            H (float): The observation matrix.
            Q (float): The process noise covariance.
            R (float): The measurement noise covariance.

        Returns:
            tuple: A tuple containing the updated state and updated covariance.
        """
        predicted_state = (
            (F * initial_state) if (initial_state != 0) else (F + initial_state)
        )
        predicted_covariance = F * initial_covariance * F + Q
        if predicted_state > 1.03:
            predicted_state = 1.03
        if predicted_state < 0.0:
            predicted_state = 0.0
        kalman_gain = (
            predicted_covariance * H / (H * predicted_covariance * H + R)
        )
        if predicted_state == 0.0:
            kalman_gain = 0.0
        updated_state = predicted_state + kalman_gain * (
            measurement - H * predicted_state
        )
        updated_covariance = (
            np.eye(1) - kalman_gain * H
        ) * predicted_covariance
        return updated_state, updated_covariance

    def eta_IR_model(self):
        if self.model == 9:
            ecirs = np.array(
                [
                    265.73504446,
                    193.25749572,
                    150.52758876,
                    153.54218556,
                    140.38102248,
                    115.90546795,
                    116.98004004,
                    135.97963103,
                    110.75206294,
                    115.14391246,
                    107.45491723,
                    103.39265128,
                    98.62933022,
                    98.20623456,
                    100.15878339,
                    93.95702161,
                    106.73098232,
                    84.60573467,
                    81.30638019,
                    92.08460278,
                ]
            )
            ejs = np.array(
                [
                    0.57213757,
                    0.46832622,
                    0.36798205,
                    0.36796971,
                    0.3456263,
                    0.28929404,
                    0.30546796,
                    0.3207472,
                    0.30742891,
                    0.27582422,
                    0.27833251,
                    0.27273583,
                    0.2688376,
                    0.26891065,
                    0.25840953,
                    0.26035265,
                    0.28407555,
                    0.2511478,
                    0.23783838,
                    0.2442877,
                ]
            )
            edirs = np.array(
                [
                    144.13570554,
                    133.383085,
                    146.844718,
                    163.665517,
                    143.11480486,
                    159.19511152,
                    171.879356,
                    172.575439,
                    182.8173,
                    199.270397,
                    216.173723,
                    245.16755,
                    237.948605,
                    271.300352,
                    321.887266,
                    297.983669,
                    339.22986,
                    515.47493,
                    713.744878,
                    1175.26785,
                ]
            )
            edjs = np.array(
                [
                    0.33382589,
                    0.108583723,
                    0.120415394,
                    0.0971390649,
                    0.17345536,
                    0.15970088,
                    0.132848596,
                    0.144802995,
                    0.102346181,
                    0.129028886,
                    0.118984009,
                    0.174626512,
                    0.198310621,
                    0.125375152,
                    0.12582753,
                    0.101837144,
                    0.144799713,
                    0.2258642,
                    0.319955204,
                    0.103599587,
                ]
            )
            c_rates = np.array(
                [
                    0.05,
                    0.1,
                    0.15,
                    0.2,
                    0.25,
                    0.3,
                    0.35,
                    0.4,
                    0.45,
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                    1.0,
                ]
            )
            dc_rates = np.array(
                [
                    1.0,
                    0.95,
                    0.9,
                    0.85,
                    0.8,
                    0.75,
                    0.7,
                    0.65,
                    0.6,
                    0.55,
                    0.5,
                    0.45,
                    0.4,
                    0.35,
                    0.3,
                    0.25,
                    0.2,
                    0.15,
                    0.1,
                    0.05,
                ]
            )

            return ecirs, ejs, edirs, edjs, c_rates, dc_rates

    def resistance_calculation(self, voltage, current, cell_num):
        # Current should be cell current i.e., current/num_cells_parallel
        R = 8.31446261815324
        T = 298.15
        F = 96485.33212
        if self.model == 9:
            ecirs, ejs, edirs, edjs, c_rates, dc_rates = self.eta_IR_model()
            if current >= 0:
                ocv = (
                    voltage
                    - (
                        (2 * R * T / F)
                        * np.arcsinh(
                            current
                            / (
                                2
                                * np.interp(
                                    (current / (self.capacity * self.soh)),
                                    c_rates,
                                    ejs,
                                )
                                * (self.capacity * self.soh)
                            )
                        )
                    )
                    - (
                        np.interp(
                            current / (self.capacity * self.soh), c_rates, ecirs
                        )
                        * 0.001
                        * (current / self.capacity * self.soh)
                    )
                )
            else:
                ocv = (
                    voltage
                    - (
                        (2 * R * T / F)
                        * np.arcsinh(
                            current
                            / (
                                2
                                * np.interp(
                                    abs(current / (self.capacity * self.soh)),
                                    dc_rates,
                                    edjs,
                                )
                                * self.capacity
                                * self.soh
                            )
                        )
                    )
                    - (
                        np.interp(
                            abs(current / (self.capacity * self.soh)),
                            dc_rates,
                            edirs,
                        )
                        * 0.001
                        * (current / (self.capacity * self.soh))
                    )
                )
        return ocv

    def process_filter(
        self,
        cumulative_time_c,
        voltages_c,
        current_c,
        interpolation=False,
        period=0.1,
    ):
        cumulative_time = pd.Series(cumulative_time_c)
        voltages = [pd.Series(volt) for volt in voltages_c]
        current = pd.Series(current_c)
        self.initial_state = [
            np.interp(voltage.iloc[0], self.y3, self.x3) for voltage in voltages
        ]
        self.initial_covariance = [1000 for _ in range(len(voltages))]

        if self.num_cells_series != len(voltages):
            raise Warning(
                "Number of cells series is not equal to the number of voltages\n Kalman Filter will be performed for available cells"
            )

        filtered_values = [self.initial_state]
        measurements = [self.initial_state]
        measures = [self.initial_state]

        if interpolation:
            time = pd.Series(np.arange(0, cumulative_time.iloc[-1], period))
            current = pd.Series(np.interp(time, cumulative_time, current))
            voltages = [
                pd.Series(np.interp(time, cumulative_time, voltage))
                for voltage in voltages
            ]
        else:
            time = cumulative_time

        time_data = time.diff().fillna(0)  # seconds
        t = 0
        kf_key = 0

        for i in range(1, len(time)):
            if abs(current.iloc[i]) <= 0.5:
                current.iloc[i] = 0.0

            measurements.append(
                [
                    self.resistance_calculation(
                        current.iloc[i] / self.num_cells_parallel,
                        voltage.iloc[i],
                        cell_num,
                    )
                    for cell_num, voltage in enumerate(voltages)
                ]
            )

            for c_no in range(len(voltages)):
                measure = []
                if self.initial_state[c_no] != 0:
                    F = 1 + (
                        ((current.iloc[i - 1]) * time_data.iloc[i - 1])
                        / (
                            (self.capacity * self.soh[c_no])
                            * 3600
                            * self.initial_state[c_no]
                        )
                    )
                else:
                    F = ((current.iloc[i - 1]) * time_data.iloc[i - 1]) / (
                        (self.capacity * self.soh[c_no]) * 3600
                    )

                if self.initial_state[c_no] != 0:
                    self.initial_state[c_no] = (
                        F * self.initial_state[c_no]
                    )  # Coulomb Counting to next SOC
                else:
                    self.initial_state[c_no] += F

                if self.model == 9:
                    if (
                        measurements[-1][c_no] < 0.2
                        or measurements[-1][c_no] > 0.999
                    ) or (
                        (
                            measurements[-1][c_no] > 0.6
                            and measurements[-1][c_no] < 0.65
                        )
                        and (0.55 < self.initial_state[c_no] < 0.7)
                    ):
                        if 0.6 < measurements[-1][c_no] < 0.65:
                            kf_key += abs(current.iloc[i] / 100) * (
                                time_data.iloc[i] / 3600
                            )

                        if (
                            measurements[-1][c_no] < 0.2
                            or measurements[-1][c_no] > 0.999
                        ):
                            kf_key = 0
                            measurement = measurements[i][c_no]
                            if kf_key < (0.01) * len(voltages):
                                (
                                    self.initial_state[c_no],
                                    self.initial_covariance[c_no],
                                ) = self.update_kalman_filter(
                                    measurement,
                                    self.initial_state[c_no],
                                    self.initial_covariance[c_no],
                                    F,
                                    self.H,
                                    self.Q,
                                    self.R,
                                )
                                measure.append(measurement)
                            else:
                                measure.append(self.initial_state[c_no])

                measures.append(measure)

            if abs(current.iloc[i]) == 0.0:
                t += time_data.iloc[i]
            else:
                t = 0

            if t > 900:
                if self.model == 9:
                    for c_no in range(len(voltages)):
                        if (
                            voltages[c_no].iloc[i] < 3.25
                            or voltages[c_no].iloc[i] > 3.35
                        ):
                            self.initial_state[c_no] = np.interp(
                                voltages[c_no].iloc[i], self.y3, self.x3
                            )
                            self.initial_covariance[c_no] = 1000
                    t = 0

            filtered_values.append(self.initial_state)

        # Transpose the list
        transposed_list = list(map(list, zip(*filtered_values)))

        # Convert each row to a pandas Series
        soc_list = [pd.Series(row) for row in transposed_list]

        return soc_list, measures
