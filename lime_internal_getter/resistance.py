import numpy as np
from . import interpolate as ip


def eta_IR_model(model):
    """
    Returns the arrays needed for model 9 internal resistance calculations.
    """
    if model == 9:
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
    if model == 3:
        edirs = np.array(
            [
                315,
                246.6443328749824,
                206.69526588300542,
                193.83939234880887,
                187.14643428767425,
                180.8131981494039,
                174.7658450399559,
                167.3501064006345,
                163.97910336754268,
                159.51354534078413,
                156.62208794982794,
                153.0108716440943,
                149.23508636207012,
                143.1705938749443,
                146.06947481040604,
                144.46026589916838,
                135.4204603057264,
                140.82055192981346,
                127.45927535999786,
                124.18127478380585,
            ]
        )
        edjs = np.array(
            [
                0.12890112470591858,
                0.15761517578042164,
                0.08377578341813284,
                0.07187502162691534,
                0.07044679408950937,
                0.10149115388869123,
                0.09168808822073447,
                0.09057498912963492,
                0.08512090338967804,
                0.09294291796200087,
                0.11182766660952792,
                0.11772096518175065,
                0.13288420014519264,
                0.12586723107830425,
                0.08653461667470366,
                0.08710358939989729,
                0.0919802722660857,
                0.08591508072160256,
                0.12423848223291523,
                0.10124370509197245,
            ]
        )
        dc_rates = np.array(
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
        return None, None, edirs, edjs, None, dc_rates
    return None, None, None, None, None, None


def resistance_calculation(params, voltage, current, cell_num, soh, model):
    """
    Calculates the OCV using an IR model approach for model=9.
    """
    # Current should be cell current i.e., current / num_cells_parallel
    R_const = np.float32(8.31446261815324)
    T = np.float32(298.15)
    F_const = np.float32(96485.33212)

    if model == 9:
        ecirs, ejs, edirs, edjs, c_rates, dc_rates = eta_IR_model(model)
        if ecirs is None:
            raise TypeError("No eta_IR model avaiable for this model type")

        capacity = params["capacity"]
        # Ensure we don't go out of range
        if current >= 0:
            ocv = (
                voltage
                - (
                    (2 * R_const * T / F_const)
                    * np.arcsinh(
                        current
                        / (
                            2
                            * ip.interp(
                                (current / (capacity * soh[cell_num])),
                                c_rates,
                                ejs,
                            )
                            * (capacity * soh[cell_num])
                        )
                    )
                )
                - (
                    ip.interp(current / (capacity * soh[cell_num]), c_rates, ecirs)
                    * 0.001
                    * (current / capacity * soh[cell_num])
                )
            )
        else:
            ocv = (
                voltage
                - (
                    (2 * R_const * T / F_const)
                    * np.arcsinh(
                        current
                        / (
                            2
                            * ip.interp(
                                abs(current / (capacity * soh[cell_num])),
                                dc_rates,
                                edjs,
                            )
                            * capacity
                            * soh[cell_num]
                        )
                    )
                )
                - (
                    ip.interp(
                        abs(current / (capacity * soh[cell_num])),
                        dc_rates,
                        edirs,
                    )
                    * 0.001
                    * (current / (capacity * soh[cell_num]))
                )
            )
        return ocv
    if model == 3:
        ecirs, ejs, edirs, edjs, c_rates, dc_rates = eta_IR_model(model)
        capacity = params["capacity"]
        ocv = (
            voltage
            - (
                (2 * R_const * T / F_const)
                * np.arcsinh(
                    current
                    / (
                        2
                        * ip.interp(
                            abs(current / (capacity * soh[cell_num])),
                            dc_rates,
                            edjs,
                        )
                        * capacity
                        * soh[cell_num]
                    )
                )
            )
            - (
                ip.interp(
                    abs(current / (capacity * soh[cell_num])),
                    dc_rates,
                    edirs,
                )
                * 0.001
                * (current / (capacity * soh[cell_num]))
            )
        )
        return ocv
    return voltage
