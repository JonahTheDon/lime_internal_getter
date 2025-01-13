import numpy as np
import pandas as pd
from parameters import get_model_9_params
from resistance import resistance_calculation
import soh


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
            self.__soh_value = [1.0 for i in range(self.num_cells_series)]
            self.__kf_key = [0.0 for i in range(self.num_cells_series)]
            self.__recal_flag = False

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
        updated_covariance = (1 - kalman_gain * H) * predicted_covariance
        return updated_state, updated_covariance

    def filter_conditions(self, current, measurement, cell_num, time_data):
        """
        Use this function to define the conditions for the Kalman Filter to be applied for each model type
        measurement=res,
        current=current.iloc[i],
        time_data=time_data.iloc[i]

        """
        if self.model == 9:
            if (measurement < 0.2 or measurement > 0.999) or (
                (measurement > 0.6 and measurement < 0.65)
                and (0.55 < self.initial_state[cell_num] < 0.7)
            ):
                if 0.6 < measurement < 0.65:
                    self.__kf_key[cell_num] += abs(current / self.capacity) * (
                        time_data / 3600
                    )

                if measurement < 0.2 or measurement > 0.999:
                    self.__kf_key[cell_num] = 0
                    measurement = measurement
                    if self.__kf_key[cell_num] < (0.005):
                        return True
                    else:
                        return False
            else:
                return False

    def __model_recalibrator(self, voltages):
        if self.model == 9:
            for c_no in range(len(voltages)):
                if voltages[c_no] < 3.25 or voltages[c_no] > 3.35:
                    self.initial_state[c_no] = float(
                        np.interp(voltages[c_no], self.y3, self.x3)
                    )
                    self.initial_covariance[c_no] = 1000

    def process_filter(
        self,
        cumulative_time_c,
        voltages_c,
        current_c,
        interpolation=False,
        period=0.1,
        soh_value=None,
        tune_parameters=None,
    ):
        cumulative_time = pd.Series(cumulative_time_c)
        voltages = [pd.Series(volt) for volt in voltages_c]
        current = pd.Series(current_c)
        volts = np.array(voltages).T
        self.initial_state = [
            np.interp(voltage.iloc[0], self.y3, self.x3) for voltage in voltages
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
        measures = [self.initial_state]
        full_measurements = [self.initial_state]
        sohs = [self.__soh_value]
        if tune_parameters:
            self.Q = tune_parameters[0]
            self.R = tune_parameters[1]

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
        for i in range(1, len(time)):
            if abs(current.iloc[i]) <= 0.5:
                current.iloc[i] = 0.0
            measure = []
            resistances = []
            for c_no in range(len(voltages)):
                res = np.interp(
                    resistance_calculation(
                        {
                            "capacity": self.capacity,
                            "num_cells_parallel": self.num_cells_parallel,
                        },
                        float(voltages[c_no].iloc[i]),
                        float(current.iloc[i]) / self.num_cells_parallel,
                        c_no,
                        self.__soh_value,
                        self.model,
                    ),
                    self.y3,
                    self.x3,
                )
                if self.initial_state[c_no] != 0:
                    self.capacity = float(self.capacity)
                    self.__soh_value[c_no] = float(self.__soh_value[c_no])
                    current.iloc[i - 1] = float(current.iloc[i - 1])
                    time_data.iloc[i - 1] = float(time_data.iloc[i - 1])
                    F = 1 + (
                        ((current.iloc[i - 1]) * time_data.iloc[i - 1])
                        / (
                            (self.capacity * self.__soh_value[c_no])
                            * 3600
                            * self.initial_state[c_no]
                        )
                    )
                else:
                    F = ((current.iloc[i - 1]) * time_data.iloc[i - 1]) / (
                        (self.capacity * self.__soh_value[c_no]) * 3600
                    )

                if self.initial_state[c_no] != 0:
                    self.initial_state[c_no] = (
                        F * self.initial_state[c_no]
                    )  # Coulomb Counting to next SOC
                else:
                    self.initial_state[c_no] += F
                resistances.append(res)
                if self.model == 9:
                    # if False: # Enable this for coloumb counting only
                    if self.filter_conditions(
                        current.iloc[i], res, c_no, time_data.iloc[i]
                    ):
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
                        )
                        measure.append(res)
                    else:
                        measure.append(self.initial_state[c_no])

            measures.append(measure.copy())
            full_measurements.append(resistances.copy())
            if abs(current.iloc[i]) == 0.0:
                t += time_data.iloc[i]
            else:
                t = 0

            if t > 900:
                self.__recal_flag = True
                self.__model_recalibrator(volts[i])
                t = 0

            self.__soh_value = soh_processor.soh_estimator(
                current.iloc[i],
                time_data.iloc[i],
                volts[i],
                self.__recal_flag,
                self.initial_state,
            )

            if t < 900:
                self.__recal_flag = False
            filtered_values.append(self.initial_state.copy())
            sohs.append(self.__soh_value.copy())
        filtered_values = list(map(list, zip(*filtered_values)))
        measures = list(map(list, zip(*measures)))
        full_measurements = list(map(list, zip(*full_measurements)))
        sohs = list(map(list, zip(*sohs)))
        self.__soh_value = [1.0 for i in range(self.num_cells_series)]
        self.__kf_key = [0.0 for i in range(self.num_cells_series)]
        self.recal_flag = False
        self.soc = filtered_values
        self.used_measurements = measures
        self.measurements = full_measurements
        self.soh = sohs


# if __name__ == "__main__":
#     filter=KalmanFilter(model=9)
#     filter.process_filter(voltages_c=[[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]], current_c=[1,2,3,4,5,6,7,8,9,10],cumulative_time_c=[1,2,3,4,5,6,7,8,9,10])
#     print(filter.soc)
#     print(filter.soh)
