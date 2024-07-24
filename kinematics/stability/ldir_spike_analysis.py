import matplotlib as mpl
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
    # plt.plot(leftcop)
    # plt.plot(rightcop)
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

    lcop_points = leftcop[xcom_peaks]
    rcop_points = rightcop[xcom_troughs]

    for i in range(len(xcom_peaks)):
        # Getting window between peak values
        xcom_index = xcom_peaks[i]
        cop_point = lcop_points[i]

        # Getting non-nan values from region

        # Making sure we are actually grabbing the last meaningful region of center of pressure
        lmos = xcom[xcom_index] - cop_point
        # print(f"L COP {cop_point}")
        # print(f"xCoM {xcom[xcom_index]}")
        lmos_values = np.append(lmos_values, lmos)

    for i in range(len(xcom_troughs)):
        # Getting window between peak values
        xcom_index = xcom_troughs[i]
        cop_point = rcop_points[i]

        # Getting non-nan values from region
        rmos = cop_point - xcom[xcom_index]
        # print(f"R COP {cop_point}")
        # print(f"xCoM {xcom[xcom_index]}")
        rmos_values = np.append(rmos_values, rmos)

    return lmos_values, rmos_values, xcom_peaks, xcom_troughs


def main():

    mouse_number = 5
    ldir = pd.read_csv(
        f"./lr-walking/ldir/lwalk-{mouse_number}.txt", delimiter=",", header=0
    )
    # print(ldir.head(5))

    # Some things to set for plotting/saving
    manual_analysis = False
    save_auto = True
    lmos_filename = f"./lr-walking/ldir/lwalk-{mouse_number}-lmos.csv"
    rmos_filename = f"./lr-walking/ldir/lwalk-{mouse_number}-rmos.csv"
    figure_title = f"Measurement of Stability For LR Walking WT {mouse_number}"
    figure_filename = f"./lr-walking/ldir/lwalk-{mouse_number}.svg"

    # Grabbing individual channels
    xcom = ldir["v1 xCoMy"].to_numpy(dtype=float)
    leftcop = ldir["v3 Left CoP"].to_numpy(dtype=float)
    rightcop = ldir["v2 Right CoP"].to_numpy(dtype=float)
    left_DS = ldir["v3 Left CoP"].to_numpy(dtype=float)
    right_DS = ldir["v2 Right CoP"].to_numpy(dtype=float)

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
        width_threshold=60,
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

    # plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8.5, 11)
    fig.tight_layout()
    # plt.savefig("./dtr-mos-output.pdf", dpi=300)

    # Saving results
    if manual_analysis is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        plt.savefig(figure_filename, dpi=300)
        print("mos results saved!")
    elif manual_analysis is False and save_auto is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        plt.savefig(figure_filename, dpi=300)
        print("mos results saved!")
    else:
        print("mos results not saved")

    plt.show()


if __name__ == "__main__":
    main()
