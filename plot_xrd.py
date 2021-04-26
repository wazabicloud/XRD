import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

dataset = [
    {
        "code": "00-037-0465",
        "label": "Zn$_3$(PO$_4$)$_2 \cdot$4H$_2$O",
        "c": "orange"
    },
    {
        "code": "00-039-0079",
        "label": "Zn$_3$(PO$_4$)$_2 \cdot$4H$_2$O",
        "c": "orange"
    },
    {
        "code": "00-039-0080",
        "label": "Zn$_3$(PO$_4$)$_2 \cdot$4H$_2$O",
        "c": "orange"
    },
    {
        "code": "00-033-1474",
        "label": "Zn$_3$(PO$_4$)$_2 \cdot$4H$_2$O",
        "c": "orange"
    },
    {
        "code": "00-010-0333",
        "label": "Zn$_3$(PO$_4$)$_2 \cdot$2H$_2$O",
        "c": "#ceb301"
    },
    {
        "code": "00-030-1491",
        "label": "Zn$_3$(PO$_4$)$_2 \cdot$2H$_2$O",
        "c": "#ceb301"
    },
    {
        "code": "00-001-1238",
        "label": "Zn",
        "c": "gray"
    },
    {
        "code": "98-024-7155",
        "label": "Zn",
        "c": "gray"
    },
    {
        "code": "98-065-3505",
        "label": "Zn",
        "c": "gray"
    },
    {
        "code": "98-067-1150",
        "label": "Zn",
        "c": "gray"
    },
    {
        "code": "98-018-0969",
        "label": "Fe",
        "c": "red"
    },
    {
        "code": "01-085-1410",
        "label": "Fe",
        "c": "red"
    },
    {
        "code": "00-001-1262",
        "label": "Fe",
        "c": "red"
    },
]

#Parametri importanti
prom = 10 #Prominence per riconoscimento picchi
h = 0.5  #Altezza relativa dei picchi da considerare

#Dati misura
def xrd_plot(csv_data, txt_peak_list, mode="normal"):

    #Estrazione punti dello spettro
    with open(csv_data, "r") as handle:
        lines_list = handle.readlines()

        #Trovo inizio dati
        for i in range(len(lines_list)):

            if "Angle" in lines_list[i]:
                start = i + 1
                break

        #Sistemo dati
        for i in range(len(lines_list)):
            if i < start:
                continue
            else:
                lines_list[i] = lines_list[i].rstrip().split(",")

                for j in range(len(lines_list[i])):
                    lines_list[i][j] = float(lines_list[i][j])

        #Creo dataframe
        df = pd.DataFrame(lines_list[start:], columns=["Angle", "Intensity"])

        max_int = 0.01*df["Intensity"].max()

        df.loc[:,"Intensity"] = df["Intensity"].div(max_int)

    if not mode == "normal":

        peaks, _ = find_peaks(df["Intensity"], prominence=prom)

        peak_width = peak_widths(df["Intensity"], peaks, rel_height=h)

        min_max_list = np.stack((peak_width[2].astype(int), peak_width[3].astype(int)), axis=-1).tolist()

        baseline_df = df

        for item in min_max_list:
            baseline_df = baseline_df.loc[(baseline_df.index < item[0]) | (baseline_df.index > item[1])]

        # def base_fit_func(x, a, b, c, d, e, f):
        #     return a*(x**5) + b*(x**4) + c*(x**3) + d*(x**2) + e*x + f

        def base_fit_func(x, a, b, c):
            return (a * np.exp(-b*x)) + c

        popt, pcov = curve_fit(base_fit_func, baseline_df["Angle"], baseline_df["Intensity"], method="lm")

        df.loc[:, "Corrected Intensity"] = df["Intensity"].sub(base_fit_func(df["Angle"], *popt)).rolling(3, center=True).mean().mul(max_int)

        if df["Corrected Intensity"].min() < 0:
            df.loc[:, "Corrected Intensity"] = df["Corrected Intensity"].sub(df["Corrected Intensity"].min())

    #Estraggo i picchi
    with open(txt_peak_list, "r") as peak_handle:
        peak_lines_list = peak_handle.readlines()

        #Pulizia dati
        for i in range(len(peak_lines_list)):
            peak_lines_list[i] = peak_lines_list[i].split("\t")

            if peak_lines_list[i][-1] == "\n":
                peak_lines_list[i][-1] = None
            else:
                peak_lines_list[i][-1] = peak_lines_list[i][-1].rstrip().replace(" ", "").split(",")

        #Creazione df dei picchi
        entry_list = []

        for i in range(len(peak_lines_list)):
            if peak_lines_list[i][-1] == None:
                continue

            for j in range(len(peak_lines_list[i][-1])):

                angle = float(peak_lines_list[i][0])

                if not mode == "normal":

                    angle_id = df["Angle"].sub(angle).abs().idxmin()

                    height = df[df.index == angle_id]["Corrected Intensity"] + 500

                else:
                    height = float(peak_lines_list[i][1]) + 1200

                code = peak_lines_list[i][-1][j]

                entry_list.append([angle - float(peak_lines_list[i][2]), height, code])

        peak_df = pd.DataFrame(entry_list, columns=["Angle", "Height", "Code"])


    #Plot del grafico, se selezionato diagnostic creo diversi grafici
    if mode == "diagnostic":
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(df["Angle"], df["Intensity"], c="black", lw=1)
        ax1.scatter(df["Angle"][peaks], df["Intensity"][peaks], c="yellow", edgecolors="black", s=20)
        ax1.scatter(df["Angle"][peak_width[2].astype(int)], df["Intensity"][peak_width[2].astype(int)], c="red", edgecolors="black", s=20)
        ax1.scatter(df["Angle"][peak_width[3].astype(int)], df["Intensity"][peak_width[3].astype(int)], c="cyan", edgecolors="black", s=20)

        ax2.plot(df["Angle"], df["Intensity"], c="black", lw=1)
        ax2.plot(df["Angle"], base_fit_func(df["Angle"], *popt), c="red", lw=1)
        ax2.scatter(baseline_df["Angle"], baseline_df["Intensity"], c="blue", s=10)

    else:
        if mode == "normal":
            plt.plot(df["Angle"], df["Intensity"], c="black", lw=1)

        elif mode == "baseline":
            plt.plot(df["Angle"], df["Corrected Intensity"], c="black", lw=1)

        #Plot degli indicatori dei picchi in base alla specie chimica a cui appartengono
        for code in peak_df["Code"].unique().tolist():
            sub_df = peak_df[peak_df["Code"] == code]

            for entry in dataset:
                if entry["code"] == code:
                    plt.scatter(sub_df["Angle"], sub_df["Height"], edgecolors="black", c=entry["c"], label=entry["label"], s=40)

        plt.xlabel(r"2$\theta$ [°]")
        plt.ylabel("Intensity")

        plt.tick_params(which="both", direction="in")
        plt.legend(frameon = False)

    plt.show()

xrd_plot("M.csv", "M.txt", mode="baseline")
xrd_plot("40.csv", "40.txt", mode="baseline")
#xrd_plot("60.csv", "60.txt", mode="diagnostic")