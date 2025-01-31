import numpy as np
import pandas as pd
from .parameters import get_model_9_params
from .resistance import resistance_calculation
from . import soh
from . import interpolate as ip


class KalmanFilter:
    def __init__(self, model):
        self.model = model
        if self.model == 9:  # GangFeng 100Ah LFP parameters
            # Load parameters for model=9
            params_9 = get_model_9_params()
            self.Q = params_9["Q"]
            self.H = params_9["H"]
            self.R = params_9["R"]
            self.x3 = params_9["x3"]
            self.y3 = params_9["y3"]
            self.capacity = params_9["capacity"]
            self.num_cells_parallel = params_9["num_cells_parallel"]
            self.num_cells_series = params_9["num_cells_series"]
            self.__soh_value = [1.0 for i in range(int(self.num_cells_series))]
            self.__kf_key = [0.0 for i in range(int(self.num_cells_series))]
            self.__recal_flag = False

    def update_kalman_filter(
        self,
        measurement,
        initial_state,
        initial_covariance,
        F,
        H,
        Q,
        R,
        filtercond,
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
        kalman_gain = predicted_covariance * H / (H * predicted_covariance * H + R)
        if predicted_state > 1.03:
            predicted_state = 1.03
        if predicted_state < 0.0:
            predicted_state = 0.0
        if predicted_state == 0.0:
            kalman_gain = 0.0
        updated_covariance = (1 - kalman_gain * H) * predicted_covariance
        if not filtercond:
            return predicted_state, updated_covariance
        updated_state = predicted_state + kalman_gain * (
            measurement - H * predicted_state
        )
        return updated_state, updated_covariance

    def filter_conditions(self, current, measurement, cell_num, time_data):
        """
        Use this function to define the conditions for the Kalman Filter to be applied for each model type
        measurement=res,
        current=current.iloc[i],
        time_data=time_data.iloc[i]

        """
        if self.model == 9:
            if ((0.6 < measurement) and (measurement < 0.65)) or (measurement > 1.01):
                self.__kf_key[cell_num] += abs(current / self.capacity) * (
                    time_data / 3600
                )
            if measurement < 0.2:
                return True

            elif (
                (measurement > 0.6 and measurement < 0.65)
                and (0.55 < self.initial_state[cell_num] < 0.7)
            ) or ((measurement > 1.01) and (self.initial_state[cell_num] > 1.0)):
                if self.__kf_key[cell_num] > (0.0):
                    return True
                else:
                    return False
            else:
                self.__kf_key[cell_num] = 0.0
                return False

    def __model_recalibrator(self, voltages):
        if self.model == 9 or self.model == 6:
            reflag = True
            for ce in voltages:
                if not ((ce < 3.25) and (ce > 3.35)):
                    reflag = False
                    break
            if reflag:
                self.__recal_flag = True
        else:
            self.__recal_flag = True
        if self.__recal_flag:
            for c_no in range(len(voltages)):
                self.initial_state[c_no] = np.float32(
                    ip.interp(voltages[c_no], self.y3, self.x3)
                )
                self.initial_covariance[c_no] = 1000

    def process_filter(
        self,
        pim_df,
        interpolation=False,
        period=0.1,
        soh_value=None,
        tune_parameters=None,
    ):
        voltages_c = [pim_df.iloc[:, p_col] for p_col in range(2, pim_df.shape[1])]
        current_c = pim_df.iloc[:, 1]
        cumulative_time_c = pim_df.iloc[:, 0]
        cumulative_time = np.array(cumulative_time_c)
        voltages = [np.array(volt) for volt in voltages_c]
        current = np.array(current_c)
        volts = np.array(voltages).T
        self.initial_state = [
            ip.interp(voltage[0], self.y3, self.x3) for voltage in voltages
        ]
        self.initial_covariance = [1000 for _ in range(len(voltages))]

        if self.num_cells_series != len(voltages):
            print(
                "Number of cells series is not equal to the number of voltages\n Kalman Filter will be performed for available cells"
            )
            self.num_cells_series = len(voltages)
            self.__soh_value = [1.0 for i in range(self.num_cells_series)]
            self.__kf_key = [0.0 for i in range(self.num_cells_series)]
        if soh_value:
            self.__soh_value = [soh_value for i in range(self.num_cells_series)]
        soh_processor = soh.SOHEstimator(
            self.num_cells_series,
            self.num_cells_parallel,
            self.capacity,
            soh=soh_value,
        )

        filtered_values = [self.initial_state.copy()]
        measures = [self.initial_state.copy()]
        full_measurements = [self.initial_state.copy()]
        sohs = [self.__soh_value.copy()]
        if tune_parameters:
            self.Q = tune_parameters[0]
            self.R = tune_parameters[1]

        if interpolation:
            time = np.array(np.arange(0, cumulative_time[-1], period))
            current = np.array(ip.interp(time, cumulative_time, current))
            voltages = [
                np.array(ip.interp(time, cumulative_time, voltage))
                for voltage in voltages
            ]
        else:
            time = cumulative_time

        time_data = np.diff(time, prepend=0)  # seconds
        t = 0
        for i in range(1, len(time)):
            if abs(current[i]) <= 0.5:
                current[i] = 0.0
            measure = []
            resistances = []
            for c_no in range(len(voltages)):
                res = ip.interp(
                    resistance_calculation(
                        {
                            "capacity": self.capacity,
                            "num_cells_parallel": self.num_cells_parallel,
                        },
                        np.float32(voltages[c_no][i]),
                        np.float32(current[i]) / self.num_cells_parallel,
                        c_no,
                        self.__soh_value,
                        self.model,
                    ),
                    self.y3,
                    self.x3,
                )
                if self.initial_state[c_no] != 0:
                    self.capacity = np.float32(self.capacity)
                    self.__soh_value[c_no] = np.float32(self.__soh_value[c_no])
                    current[i - 1] = np.float32(current[i - 1])
                    time_data[i - 1] = np.float32(time_data[i - 1])
                    F = 1 + (
                        ((current[i - 1]) * time_data[i - 1])
                        / (
                            (self.capacity * self.__soh_value[c_no])
                            * 3600
                            * self.initial_state[c_no]
                        )
                    )
                else:
                    F = ((current[i - 1]) * time_data[i - 1]) / (
                        (self.capacity * self.__soh_value[c_no]) * 3600
                    )
                filtercond = self.filter_conditions(current[i], res, c_no, time_data[i])
                resistances.append(res)
                (
                    self.initial_state[c_no],
                    self.initial_covariance[c_no],
                ) = self.update_kalman_filter(
                    res,
                    self.initial_state[c_no],
                    self.initial_covariance[c_no],
                    F,
                    self.H,
                    self.Q,
                    self.R,
                    filtercond,
                )

                if filtercond:
                    measure.append(res)
                else:
                    measure.append(self.initial_state[c_no])

            measures.append(measure.copy())
            full_measurements.append(resistances.copy())
            if abs(current[i]) == 0.0:
                t += time_data[i]
            else:
                t = 0

            if t > 900:
                self.__model_recalibrator(volts[i])
                t = 0

            self.__soh_value = soh_processor.soh_estimator(
                current[i],
                time_data[i],
                volts[i],
                self.__recal_flag,
                self.initial_state,
            )

            if t < 900:
                self.__recal_flag = False
            filtered_values.append(self.initial_state.copy())
            sohs.append(self.__soh_value.copy())
            if self.__recal_flag:
                self.__recal_flag = False
        filtered_values = list(map(list, zip(*filtered_values)))
        measures = list(map(list, zip(*measures)))
        full_measurements = list(map(list, zip(*full_measurements)))
        sohs = list(map(list, zip(*sohs)))
        self.__soh_value = [1.0 for i in range(int(self.num_cells_series))]
        self.__kf_key = [0.0 for i in range(int(self.num_cells_series))]
        self.recal_flag = False
        self.soc = [pd.Series(fil) for fil in filtered_values]
        self.used_measurements = [pd.Series(mes) for mes in measures]
        self.measurements = [pd.Series(mes) for mes in full_measurements]
        self.soh = [pd.Series(s) for s in sohs]
