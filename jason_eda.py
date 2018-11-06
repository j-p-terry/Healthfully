import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

tick_locs = [52*i for i in range(9)]
x_ticks = [str(2010 + i) for i in range(9)]

# Census population in 2010 (millions)
pop_2010 = 308.7

data_path = "/Users/jasonterry/Dropbox/Healthfully/Clean Data/" \
            "weather_flu_google_9.csv"
save_path = "/Users/jasonterry/Documents/Scripts/DSI/" \
            "proj1_plots/final_eda/"

searches = ["flu", "cough_medicine", "flu_clinic", "flu_shot",
            "flu_vaccine", "flu", "influenza", "tamiflu"]

rates = ["all_rate", "rate_0_4", "rate_5_17", "rate_18_49", "rate_50_64",
         "rate_65_plus"]

data_columns = ["total.specimens", "percent.positive", "flu",
                "median_tmax", "median_tmin", "median_t", "total_prcp",
                "cough_medicine", "flu_clinic", "flu_shot", "flu_vaccine",
                "flu", "influenza",	"oseltamivir", "relenza", "tamiflu",
                "zanamivir", "cases", "humidity", "flu_symptoms",
                "vac_num", "vac_effectiveness", "all_rate", "rate_0_4",
                "rate_5_17", "rate_18_49", "rate_50_64", "rate_65_plus",
                "absolute_humidity"]

corr_columns = ["total.specimens_z", "percent.positive_z",  "median_t_z",
                "total_prcp_z", "oseltamivir_z", "cough_medicine_z",
                "flu_clinic_z", "flu_shot_z", "flu_vaccine_z",
                "flu_z", "influenza_z",	"tamiflu_z", "cases", "year",
                "season", "humidity_z", "flu_symptoms_z",
                "vac_num_z", "vac_effectiveness_z","all_rate_z",
                "rate_0_4_z", "rate_5_17_z", "rate_18_49_z",
                "rate_50_64_z", "rate_65_plus_z", "absolute_humidity_z"]

stat_columns = ["total.specimens_z", "percent.positive_z", "cases_z",
                "tamiflu_z", "vac_effectiveness_z", "vac_num_z"]

X_cols = ["total.specimens", "percent.positive",  "median_t",
          "total_prcp", "oseltamivir", "cough_medicine",
          "flu_clinic", "flu_shot", "flu_vaccine",
          "flu", "influenza",	"tamiflu", "year", "humidity",
          "flu_symptoms", "vac_rate", "vac_effectiveness", "all_rate",
          "rate_0_4", "rate_5_17", "rate_18_49", "rate_50_64",
          "rate_65_plus", "absolute_humidity"]

y_col = "cases"


def corr_plots(data, pair=False):

    """Makes correlation with the option of pair plots too"""

    for column in data_columns:

        data[column + "_z"] = (data[[column]] - data[[column]].mean()) / \
                              data[[column]].std()
        data[column + "_z"] = (data[[column]] - data[[column]].mean()) / \
                              data[[column]].std()

    plt.close()

    data = data[data["season"]<2017]

    corr_data = data[corr_columns]
    """Make correlation matrix"""
    corr=corr_data.corr()
    plt.figure(figsize=(17, 14))
    sb.heatmap(corr, xticklabels=corr.columns.values,
               yticklabels=corr.columns.values, cmap='bwr')
    plt.savefig(save_path + "corr_matrix7.pdf")
    plt.close()

    if pair:
        """Make pair plots"""
        sb.pairplot(corr_data)
        plt.savefig(save_path + "pair_plot7.png")
        plt.close()


def year_plots(data, save=False, col="cases", all=False):

    """Makes violin plots for data per year"""

    plt.close()

    if not all:
        data = data[data['season'] < 2017]
    else:
        data = data[data["season"] < 2018]

    months = {"January": 5, "February": 8, "March": 13, "April": 17,
              "May": 22, "June": 26, "July": 31, "August": 35,
              "September": 39, "October": 44, "November": 48,
              "December": 53}

    names = list(months.keys())

    years = [i for i in range(2010, 2017)]
    if all:
        years.append(2017)

    stat_dic = {}
    for key in months:
        stat_dic[key] = [0 for i in range(2010, 2017)]
        if all:
            stat_dic[key].append(2017)

    for index, row in data.iterrows():

        if row["week"] < 6:
            month = "January"
        else:
            for i in range(len(names)):
                if months[names[i - 1]] < row["week"] <= months[names[i]]:
                    month = names[i]
        if col != "percent.positive":
            stat_dic[month][years.index(int(row["season"]))] += row[col]
        else:
            stat_dic[month][years.index(int(row["season"]))] += \
                row[col]/months[month]
        if col == "cases" or col == "total.specimens":
            stat_dic[month][years.index(int(row["season"]))] /= 1e4
        elif col == 'median_t':
            stat_dic[month][years.index(int(row["season"]))] /= 2.0

    if not all:
        year_dic = {2010 + i: np.array([]) for i in range(7)}
    else:
        year_dic = {2010 + i: np.array([]) for i in range(8)}
    for month in names:
        for i in range(len(list(year_dic))):
            year = list(year_dic.keys())[i]
            year_dic[year] = np.append(year_dic[year], stat_dic[month][i])
    #
    stat_data = np.array([stat_dic[year] for year in stat_dic])


    """Violin"""
    fig, axes = plt.subplots(nrows=1, figsize=(14, 9))#, ncols=2)

    ticks = [i + 1 for i in range(7)]
    if all:
        ticks.append(8)

    axes.set_xlabel("Season")
    if col == "cases":
        axes.set_ylabel("Cases (tens of thousands)")
    elif col == "median_t":
        axes.set_ylabel(col.title() + r" ($\degree$F)")
    elif col in rates:
        axes.set_ylabel(col.title() + " per 100,000")
    elif col == "absolute_humidity":
        axes.set_ylabel(col.title() + r" $(g/m^{3})$")
    elif col in searches:
        axes.set_ylabel(col.title() + " (arbitrary units)")
    else:
        axes.set_ylabel(col.title())

    axes.violinplot(stat_data, showmedians=True)

    axes.yaxis.grid(True, color='w')

    plt.setp(axes, xticks=ticks, xticklabels=x_ticks[:-1])

    if save:
        if not all:
            plt.savefig(save_path + str(col) + "_train_violin_total_yr8.pdf")
            print("saved at " + save_path + str(col) +
                  "_train_violin_total_yr8.pdf")
        else:
            plt.savefig(save_path + str(col) + "_all_violin_total_yr8.pdf")
            print("saved at " + save_path + str(col) +
                  "_all_violin_total_yr8.pdf")
        plt.close()
    else:
        plt.show()

    return stat_data



def month_plots(data, save=False, col="cases", all=False):

    """Makes violin plots for data by month"""

    plt.close()

    if not all:
        data = data[data['season'] < 2017]
    else:
        data = data[data["season"] < 2018]

    months = {"January": 5, "February": 8, "March": 13, "April": 17,
              "May": 22, "June": 26, "July": 31, "August": 35,
              "September": 39, "October": 44, "November": 48,
              "December": 53}

    names = list(months.keys())

    years = [i for i in range(2010, 2017)]
    if all:
        years.append(2017)

    stat_dic = {}
    for key in months:
        stat_dic[key] = [0 for i in range(2010, 2017)]
        if all:
            stat_dic[key].append(2017)
    for index, row in data.iterrows():

        if row["week"] < 6:
            month = "January"
        else:
            for i in range(len(names)):
                if months[names[i - 1]] < row["week"] <= months[names[i]]:
                    month = names[i]
        if col != "percent.positive":
            stat_dic[month][years.index(int(row["season"]))] += row[col]
        else:
            stat_dic[month][years.index(int(row["season"]))] += \
                row[col]/months[month]
        if col == "cases" or col == "total.specimens":
            stat_dic[month][years.index(int(row["season"]))] /= 1e4
        elif col == 'median_t':
            stat_dic[month][years.index(int(row["season"]))] /= 2.

    stat_data = [stat_dic[month] for month in months]

    """Violin d"""
    fig, axes = plt.subplots(nrows=1, figsize=(14, 9))#, ncols=2)

    ticks = [i + 1 for i in range(len(names))]

    axes.set_xlabel("Season")
    if col == "cases":
        axes.set_ylabel("Cases (tens of thousands)")
    elif col == "median_t":
        axes.set_ylabel(col.title() + r" ($\degree$F)")
    elif col in rates:
        axes.set_ylabel(col.title() + " per 100,000")
    elif col=="absolute_humidity":
        axes.set_ylabel(col.title() + r" $(g/m^{3})$")
    elif col in searches:
        axes.set_ylabel(col.title() + " (arbitrary units)")
    else:
        axes.set_ylabel(col.title())

    axes.violinplot(stat_data, showmedians=True)

    axes.yaxis.grid(True, color='w')

    plt.setp(axes, xticks=ticks, xticklabels=names)

    if save:
        if not all:
            plt.savefig(save_path + str(col) +
                        "_train_violin_total_month8.pdf")
            print("saved at " + save_path + str(col) +
                  "_train_violin_total_month8.pdf")
        else:
            plt.savefig(save_path + str(col) +
                        "_all_violin_total_month8.pdf")
            print("saved at " + save_path + str(col) +
                  "_all_violin_total_month8.pdf")
        plt.close()
    else:
        plt.show()

    return stat_data


def scatter(data):

    for column in data_columns:

        data[column + "_z"] = (data[[column]] - data[[column]].mean()) / data[[column]].std()
        data[column + "_z"] = (data[[column]] - data[[column]].mean()) / data[[column]].std()

    """Scatter plots of various pairs"""
    # plt.close()
    # plt.scatter(data["weeks_sequential"], data["cases_z"], s=3)
    # plt.scatter(data["weeks_sequential"], data["flu_shot_z"], s=3)
    #
    # plt.xticks(tick_locs, x_ticks)
    # plt.legend(loc="best")
    # plt.xlabel("Season")
    # plt.ylabel("z-score")
    # # plt.show()
    # plt.savefig(save_path + "cases_vs_flu_shot.pdf")
    # plt.close()

    plt.close()
    plt.scatter(data["weeks_sequential"], data["cases_z"], s=3)
    plt.scatter(data["weeks_sequential"], data["all_rate_z"], s=3)

    plt.xticks(tick_locs, x_ticks)
    plt.legend(loc="best")
    plt.xlabel("Season")
    plt.ylabel("z-score")
    # plt.show()
    plt.savefig(save_path + "cases_vs_all_rate.pdf")
    plt.close()

    plt.close()
    plt.scatter(data["weeks_sequential"], data["cases_z"], s=3)
    plt.scatter(data["weeks_sequential"], data["rate_0_4_z"], s=3)

    plt.xticks(tick_locs, x_ticks)
    plt.legend(loc="best")
    plt.xlabel("Season")
    plt.ylabel("z-score")
    # plt.show()
    plt.savefig(save_path + "cases_vs_rate_0_5.pdf")
    plt.close()


    plt.scatter(data["weeks_sequential"], data["cases_z"], s=3)
    plt.scatter(data["weeks_sequential"], data["rate_65_plus_z"], s=3)
    plt.xticks(tick_locs, x_ticks)
    plt.legend(loc="best")
    plt.xlabel("Season")
    plt.ylabel("z-score")
    # plt.show()
    plt.savefig(save_path + "cases_vs_65_plus.pdf")
    plt.close()
    #
    # plt.scatter(data["weeks_sequential"], data["cases_z"], s=3)
    # plt.scatter(data["weeks_sequential"], data["humidity_z"], s=3)
    # plt.xticks(tick_locs, x_ticks)
    # plt.legend(loc="best")
    # plt.xlabel("Season")
    # plt.ylabel("z-score")
    # # plt.show()
    # plt.savefig(save_path + "cases_vs_humidity.pdf")
    # plt.close()
    #
    # plt.scatter(data["weeks_sequential"], data["cases_z"], s=3)
    # plt.scatter(data["weeks_sequential"], data["flu_symptoms_z"], s=3)
    # plt.xticks(tick_locs, x_ticks)
    # plt.legend(loc="best")
    # plt.xlabel("Season")
    # plt.ylabel("z-score")
    # # plt.show()
    # plt.savefig(save_path + "cases_vs_flu_symptoms.pdf")
    # plt.close()
    #
    #
    # plt.scatter(data["weeks_sequential"], data["cases_z"], s=3)
    # plt.scatter(data["weeks_sequential"], data["tamiflu_z"], s=3)
    # plt.xticks(tick_locs, x_ticks)
    # plt.legend(loc="best")
    # plt.xlabel("Season")
    # plt.ylabel("z-score")
    # # plt.show()
    # plt.savefig(save_path + "cases_vs_tamiflu.pdf")
    # plt.close()
    #
    # #
    # plt.scatter(data["weeks_sequential"], data["cases_z"], s=3)
    # plt.scatter(data["weeks_sequential"], data["median_t_z"], s=3)
    # plt.xticks(tick_locs[:6], x_ticks[:6])
    # plt.legend(loc="best")
    # plt.xlabel("Season")
    # plt.ylabel("z-score")
    # plt.savefig(save_path + "cases_vs_median_t.pdf")
    # plt.close()
    #
    # plt.scatter(data["weeks_sequential"], data["cases_z"], s=3)
    # plt.scatter(data["weeks_sequential"], data["vac_num_z"], s=3)
    # plt.xticks(tick_locs[:6], x_ticks[:6])
    # plt.legend(loc="best")
    # plt.xlabel("Season")
    # plt.ylabel("z-score")
    # plt.savefig(save_path + "cases_vs_vac_num.pdf")
    # plt.close()
    #
    # plt.scatter(data["weeks_sequential"], data["cases_z"], s=3)
    # plt.scatter(data["weeks_sequential"],
    #             data["vac_effectiveness_z"], s=3)
    # plt.xticks(tick_locs[:6], x_ticks[:6])
    # plt.legend(loc="best")
    # plt.xlabel("Season")
    # plt.ylabel("z-score")
    # plt.savefig(save_path + "cases_vs_effectiveness.pdf")
    # plt.close()
    """End of scatter plots"""


def get_data():

    """Reads and cleans the data"""

    data = pd.read_csv(data_path)

    data["percent.positive"] = data["percent.positive"].fillna(0)
    data["total.specimens"] = data["total.specimens"].fillna(0)
    for column in ["median_t", "median_tmin", "median_tmax", "total_prcp",
                   "humidity", "all_rate", "rate_0_4",
                   "rate_5_17", "rate_18_49", "rate_50_64", "rate_65_plus"]:

        data[column] = data[column].fillna(method="ffill")

    data["cases"] = data["total.specimens"]*data["percent.positive"]

    data["weeks_sequential"] = [i for i in range(len(data))]

    seasons = np.array([])
    c1 = 0
    c2 = 0
    for index, row in data.iterrows():
        if row["week"] > 39:
            seasons = np.append(seasons, row["year"])
            c1 += 1
        else:
            seasons = np.append(seasons, row["year"] - 1)
            c2 += 1
    data["season"] = seasons
    data["vac_rate"] = data["vac_num"] / pop_2010

    return data

def run_all(all=False):

    """Runs all eda, with option for models"""
    data = get_data()


    columns = ["total.specimens", "percent.positive", "flu", "median_t",
               "total_prcp", "cough_medicine", "flu_clinic", "flu_shot",
               "flu_vaccine", "flu", "influenza", "tamiflu", "all_rate",
               "rate_0_4", "rate_5_17", "rate_18_49", "rate_50_64",
               "rate_65_plus", "absolute_humidity", "cases"]
    for column in columns:
        month_plots(data, save=True, col=column, all=all)
        year_plots(data, save=True, col=column, all=all)


if __name__ == "__main__":
    print("Running")
    run_all(all=True)
    print("Done")
