import numpy as np


class SOHEstimator:
    def __init__(
        self, num_cells_series, num_cells_parallel, capacity, soh=None
    ):
        self.num_cells_series = num_cells_series
        self.num_cells_parallel = num_cells_parallel
        self.capacity = capacity
        if soh:
            self.soh = [soh for i in range(int(self.num_cells_series))]
        else:
            self.soh = [1.0 for i in range(int(self.num_cells_series))]
        self.__prev_soh = [None for i in range(int(self.num_cells_series))]
        self.__ah1 = 0.0
        self.__ah2 = 0.0
        self.__wait_flag = 0
        self.__check_key = 0
        self.__check_key1 = False
        self.__cumu_ah = np.float32(0.0)
        self.__soc1 = [None for i in range(int(self.num_cells_series))]
        self.__soc2 = [None for i in range(int(self.num_cells_series))]

    def soh_estimator(
        self, current, time_diff, voltage, recal_flag, initial_state
    ):
        cellcurr = np.float32(current) / self.num_cells_parallel
        self.__cumu_ah = self.__cumu_ah + (cellcurr * time_diff)
        if self.__ah1 == 0.0 and self.__ah2 == 0.0 and (self.__wait_flag == 0):
            if self.__check_key == 0:
                self.__check_key = 1
            else:
                for j in range(int(self.num_cells_series)):
                    if self.__prev_soh[j]:
                        self.soh[j] = self.__prev_soh[j]
        if recal_flag:
            for j in range(int(self.num_cells_series)):
                if np.float32(voltage[j]) != 0:
                    self.__check_key1 = True
                else:
                    cellcurr = 0.0
                    self.__wait_flag = 0
                    self.__cumu_ah = 0.0
                    self.__check_key1 = False
                    self.__ah1 = 0.0
                    self.__ah2 = 0.0

            if self.__check_key1:
                if self.__wait_flag == 0:
                    self.__ah1 = self.__cumu_ah
                    for j in range(int(self.num_cells_series)):
                        self.__soc1[j] = initial_state[j]
                    self.__wait_flag = 1

                elif self.__wait_flag == 1:
                    self.__ah2 = self.__cumu_ah
                    for j in range(int(self.num_cells_series)):
                        self.__soc2[j] = initial_state[j]

                    if abs(self.__ah1 - self.__ah2) > (
                        (self.capacity * 3600) / 2
                    ):
                        for j in range(int(self.num_cells_series)):
                            self.soh[j] = abs(
                                (
                                    (self.__ah2 - self.__ah1)
                                    / (self.__soc2[j] - self.__soc1[j])
                                )
                            ) / (self.capacity * 3600)

                            self.__prev_soh[j] = self.soh[j]
                        self.__wait_flag = 0
                        self.__cumu_ah = 0.0
        return self.soh
