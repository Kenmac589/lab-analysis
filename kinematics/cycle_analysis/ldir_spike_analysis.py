import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns


def mos_marks(related_trace, leftcop, rightcop, title="Select Points"):
    """Manually annotate points of interest on a given trace
    :param related_trace: Trace you want to annotate

    :return manual_marks_x: array of indices to approx desired value in original trace
    :return manual_marks_y: array of selected values
    """

    # Removing 0 values
    rightcop = np.where(rightcop == 0.0, np.nan, rightcop)
    leftcop = np.where(leftcop == 0.0, np.nan, leftcop)

    # Correcting to DS regions are close to label
    left_adjustment = np.mean(related_trace) + 0.5
    right_adjustment = np.mean(related_trace) - 0.5

    rightcop = rightcop * right_adjustment
    leftcop = leftcop * left_adjustment

    # Open interface with trace
    plt.plot(related_trace)
    plt.plot(leftcop)
    plt.plot(rightcop)
    plt.title(title)

    # Go through and label regions desired
    manual_marks_pair = plt.ginput(0, 0)

    # Store x coordinates as rounded off ints to be used as indices
    manual_marks_x = np.asarray(list(map(lambda x: x[0], manual_marks_pair)))
    manual_marks_x = manual_marks_x.astype(np.int32)

    # Store y coordinates as the actual value desired
    manual_marks_y = np.asarray(list(map(lambda x: x[1], manual_marks_pair)))
    plt.show()

    return manual_marks_x, manual_marks_y


def mos(
    xcom, leftcop, rightcop, leftds, rightds, manual_peaks=False, width_threshold=40
):

    # Remove periods where it is not present or not valid
    # left_band = np.percentile(xcom, q=50)
    rightcop = np.where(rightcop == 0.0, np.nan, rightcop)
    leftcop = np.where(leftcop == 0.0, np.nan, leftcop)
    # rightcop[rightcop < right_band] = np.nan
    # leftcop[leftcop < left_band] = np.nan

    # Optional manual point selection
    if manual_peaks is False:
        # Getting peaks and troughs
        xcom_peaks, _ = sp.signal.find_peaks(xcom, width=width_threshold)
        xcom_troughs, _ = sp.signal.find_peaks(-xcom, width=width_threshold)
    elif manual_peaks is True:
        xcom_peaks, _ = mos_marks(xcom, leftds, rightds, title="Select Peaks")
        xcom_troughs, _ = mos_marks(xcom, leftds, rightds, title="Select Troughs")
    else:
        print("The `manual` variable must be a boolean")

    lmos_values = np.array([])
    rmos_values = np.array([])

    for i in range(len(xcom_peaks) - 1):
        # Getting window between peak values
        beginning = xcom_peaks[i]
        end = xcom_peaks[i + 1]
        region_to_consider = leftcop[beginning:end]

        # Getting non-nan values from region
        value_cop = region_to_consider[~np.isnan(region_to_consider)]

        # Making sure we are actually grabbing the last meaningful region of center of pressure
        if value_cop.shape[0] >= 2:
            cop_point = np.mean(value_cop)
            lmos = cop_point - xcom[beginning]
            lmos_values = np.append(lmos_values, lmos)

    for i in range(len(xcom_troughs) - 1):
        # Getting window between peak values
        beginning = xcom_troughs[i]
        end = xcom_troughs[i + 1]
        region_to_consider = rightcop[beginning:end]

        # Getting non-nan values from region
        value_cop = region_to_consider[~np.isnan(region_to_consider)]
        if value_cop.shape[0] >= 2:
            cop_point = np.mean(value_cop)
            rmos = xcom[beginning] - cop_point
            rmos_values = np.append(rmos_values, rmos)

    return lmos_values, rmos_values, xcom_peaks, xcom_troughs


def main():

    ldir_1 = pd.read_csv("./lr-walking/ldir/lwalk-1.txt", delimiter=",", header=0)
    print(ldir_1.head(5))

    # Some things to set for plotting/saving
    manual_analysis = False
    save_auto = False
    lmos_filename = "./lr-walking/ldir/lwalk-1-lmos.csv"
    rmos_filename = "./lr-walking/ldir/lwalk-1-rmos.csv"
    figure_title = "Measurement of Stability For LR Walking WT"

    # Grabbing individual channels
    xcom = ldir_1["v1 xCOMy"].to_numpy(dtype=float)
    leftcop = ldir_1["v3 Left CoP"].to_numpy(dtype=float)
    rightcop = ldir_1["v2 Right CoP"].to_numpy(dtype=float)
    left_DS = ldir_1["v3 Left CoP"].to_numpy(dtype=float)
    right_DS = ldir_1["v2 Right CoP"].to_numpy(dtype=float)

    # Remove periods where it is not present or not valid
    # leftcop = np.where(leftcop == 0.0, np.nan, leftcop)
    rightcop = np.where(rightcop == 0.0, np.nan, rightcop)
    leftcop = np.where(leftcop == 0.0, np.nan, leftcop)
    # right_band = 2
    # rightcop[rightcop < right_band] = np.nan
    # leftcop[leftcop < left_band] = np.nan

    lmos, rmos, xcom_peaks, xcom_troughs = mos(
        xcom,
        leftcop,
        rightcop,
        left_DS,
        right_DS,
        manual_peaks=manual_analysis,
    )

    # Plotting
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(style="white", font_scale=1.0, rc=custom_params)

    # Figure for M4 perturbation
    xcom_legend = [
        "xCoM",
        "xCoM peaks",
        "xCoM troughs",
        "L COP",
        "R COP",
    ]
    mos_legend = ["L MoS", "R MoS"]
    fig, axs = plt.subplots(2)
    fig.suptitle(figure_title)

    # For plotting figure demonstrating how calculation was done
    axs[0].set_title("How MoS is Derived")
    axs[0].plot(xcom)
    axs[0].plot(xcom_peaks, xcom[xcom_peaks], "^")
    axs[0].plot(xcom_troughs, xcom[xcom_troughs], "v")
    axs[0].plot(leftcop)
    axs[0].plot(rightcop)
    axs[0].legend(xcom_legend, bbox_to_anchor=(1, 0.7))

    # Looking at result
    axs[1].set_title("MoS Result")
    axs[1].bar(0, np.mean(lmos), yerr=np.std(lmos), capsize=5)
    axs[1].bar(1, np.mean(rmos), yerr=np.std(rmos), capsize=5)
    axs[1].legend(mos_legend, bbox_to_anchor=(1, 0.7))

    plt.tight_layout()
    # plt.savefig("./dtr-mos-output.pdf", dpi=300)
    plt.show()

    # Saving results
    if manual_analysis is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        print("mos results saved!")
    elif manual_analysis is False and save_auto is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        print("mos results saved!")
    else:
        print("mos results not saved")


if __name__ == "__main__":
    main()
