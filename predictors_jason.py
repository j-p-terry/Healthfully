import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
import sklearn.linear_model as lin

tick_locs = [52*i for i in range(9)]
x_ticks = [str(2010 + i) for i in range(9)]

# Census population in 2010 (millions)
pop_2010 = 308.7

data_path = "/Users/jasonterry/Dropbox/Healthfully/Clean Data/" \
            "weather_flu_google_9.csv"

save_path = "/Users/jasonterry/Documents/Scripts/DSI/" \
            "proj1_plots/selected/final/"

X_cols = ["median_t", "week", "cough_medicine",
          "flu_clinic", "flu_shot", "flu_vaccine",
          "flu", "influenza", "tamiflu", "humidity",
          "flu_symptoms"]

y_col = "cases"


def fit_violin(y_test, y_pred, type="linear", depth=8):

    """Makes a violin plot of the predicted vs test results"""

    fig, axes = plt.subplots(nrows=1, figsize=(14, 9))
    small=10
    medium=12
    big=14
    plt.rc('font', size=small)
    plt.rc('axes', titlesize=small)
    plt.rc('axes', labelsize=medium)
    plt.rc('xtick', labelsize=small)
    plt.rc('ytick', labelsize=small)
    plt.rc('legend', fontsize=small)
    plt.rc('figure', titlesize=big)
    axes.set_ylabel("Cases (hundreds)")

    fit_data = [y_test, y_pred]

    axes.violinplot(fit_data, showmedians=True)

    if type == "bases":
        sub_title = "Baysian Ridge Regression "
    elif type == "decision":
        sub_title = "Decision Tree (depth = " + str(depth) + ") "
    else:
        sub_title = "Linear"

    plt.title(sub_title + "Predicted 2017 cases")
    plt.setp(axes, xticks=[1, 2], xticklabels=["Test", "Predicted"])
    axes.yaxis.grid(True, color='w')

    if type == "decision":
        plt.savefig(save_path + type + "_fit_vs_pred_violin_"  +
                    str(depth) + ".png")
        print("saved at " + save_path + type + "_fit_vs_pred_violin_"  +
              str(depth) + ".png")
    else:
        plt.savefig(save_path + type + "_fit_vs_pred_violin_" + ".png")
        print("saved at " + save_path + type + "_fit_vs_pred_violin_" +
              ".png")

    plt.close()


def get_metrics(y_test, y_pred):

    """Gets the metrics of the fit"""

    if min(y_pred) < 0:
        print(min(y_pred))
        y_pred += abs(min(y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    r2 = r2.__round__(4)
    print("r^2 = " + str(r2))
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    rmse = rmse.__round__(4)
    print("rmse " + str(rmse))

    return r2, rmse


def fit_and_plot(data, linear=True, decision=True,
                 bayes=True, test_d=False):

    """Fits and plots results for specified models"""

    plt.close()

    metric = {}

    test = data[data["season"] == 2017]
    train = data[data["season"] < 2017]

    for column in list(test):
        train[column] = train[[column]].fillna(method="ffill")
        test[column] = test[[column]].fillna(method="ffill")

    y_test = test["cases"] / 1e4
    y_train = train["cases"] / 1e4
    X_train = train[X_cols]
    X_test = test[X_cols]
    print(type(X_test))
    print(type(X_train))

    if linear:
        """Linear"""
        print("Linear")
        regressor = lin.LinearRegression()
        fit = regressor.fit(X_train, y_train)
        coefficients = fit.coef_
        print("coefficients ")
        print(coefficients)
        y_pred = fit.predict(X_test)

        r2, rmse = get_metrics(y_test, y_pred)
        coefficients = fit.coef_
        print(coefficients)

        plt.figure(figsize=(14, 9))
        small=10
        medium=12
        big=14
        plt.rc('font', size=small)
        plt.rc('axes', titlesize=small)
        plt.rc('axes', labelsize=medium)
        plt.rc('xtick', labelsize=small)
        plt.rc('ytick', labelsize=small)
        plt.rc('legend', fontsize=small)
        plt.rc('figure', titlesize=big)
        plt.plot([i for i in range(len(y_test))], y_test,
                    label="Actual")
        plt.plot([i for i in range(len(y_test))], y_pred,
                    label="Predicted")

        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (hundreds)")
        plt.title("Predicted 2017 Cases Linear")

        # plt.text(40, 1.3, "RMSE = " + str(rmse))
        # plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()

        plt.savefig(save_path + "lin_predicted_cases_2017.png")
        plt.close()
        fit_violin(y_test, y_pred, type="linear")

        metric["linear"] = [r2, rmse, coefficients]

    if decision:
        """Decision tree"""
        if test_d:
            rmses = []
            r2s = []
            ds = []
            for depth in range(1, 16):
                print("decision")
                regressor = tree.DecisionTreeRegressor(max_depth=depth)
                fit = regressor.fit(X_train, y_train)
                y_pred = fit.predict(X_test)
                if min(y_pred) < 0:
                    print(min(y_pred))
                    y_pred += abs(min(y_pred))
                r2, rmse = get_metrics(y_test, y_pred)
                rmses.append(rmse)
                r2s.append(r2)
                ds.append(depth)
            plt.figure(figsize=(14, 9))
            small = 10
            medium = 12
            big = 14
            plt.rc('font', size=small)
            plt.rc('axes', titlesize=small)
            plt.rc('axes', labelsize=medium)
            plt.rc('xtick', labelsize=small)
            plt.rc('ytick', labelsize=small)
            plt.rc('legend', fontsize=small)
            plt.rc('figure', titlesize=big)
            plt.plot(ds, r2s, label=r'$R^{2}$')
            plt.plot(ds, rmses, label = "RMSE")
            plt.xlabel("Depth")
            plt.legend(loc="best")
            plt.title("Metrics versus depth")
            plt.savefig(save_path + "d_r2_rmse_v_depth.png")
            plt.close()
        else:
            depth = 8
            print("decision")
            regressor = tree.DecisionTreeRegressor(max_depth=depth)
            fit = regressor.fit(X_train, y_train)

            y_pred = fit.predict(X_test)
            if min(y_pred) < 0:
                print(min(y_pred))
                y_pred += abs(min(y_pred))

            r2, rmse = get_metrics(y_test, y_pred)
            plt.figure(figsize=(14, 9))

            small = 10
            medium = 12
            big = 14
            plt.rc('font', size=small)
            plt.rc('axes', titlesize=small)
            plt.rc('axes', labelsize=medium)
            plt.rc('xtick', labelsize=small)
            plt.rc('ytick', labelsize=small)
            plt.rc('legend', fontsize=small)
            plt.rc('figure', titlesize=big)

            plt.plot([i for i in range(len(y_test))], y_test,
                        label="Actual")
            plt.plot([i for i in range(len(y_test))], y_pred,
                        label="Predicted")
            plt.legend(loc="best")
            plt.xlabel("Season week")
            plt.ylabel("Cases (hundreds)")
            plt.title("Predicted 2017 Cases Decision Tree")

            # plt.text(40, 1.3, "RMSE = " + str(rmse))
            # plt.text(40, 1., "Depth = " + str(depth))
            # plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
            # plt.show()

            plt.savefig(save_path + "d_tree_predicted_cases_2017_d"
                        + str(depth) + ".png")
            plt.close()
            fit_violin(y_test, y_pred, type="decision", depth=depth)

            metric["decision"] = [r2, rmse, depth]

    if bayes:
        """Bayesian Ridge regression"""
        regressor = lin.BayesianRidge()

        fit = regressor.fit(X_train, y_train)
        print("bayes")
        y_pred = fit.predict(X_test)
        if min(y_pred) < 0:
            print(min(y_pred))
            y_pred += abs(min(y_pred))

        r2, rmse = get_metrics(y_test, y_pred)

        print("rmse " + str(rmse), "\n")
        plt.figure(figsize=(14, 9))
        plt.plot([i for i in range(len(y_test))], y_test,
                    label="Actual")
        plt.plot([i for i in range(len(y_test))], y_pred,
                    label="Predicted")
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (hundreds)")
        plt.title("Predicted 2017 Bayesian Ridge Regression")

        # plt.text(40, 1.3, "RMSE = " + str(rmse))
        # plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()

        plt.savefig(save_path + "brr_predicted_cases_2017.png")
        plt.close()
        fit_violin(y_test, y_pred, type="bayes")

        metric["bayes"]=[r2, rmse]

        print(metric)


def get_data():

    """Reads and cleans the data"""

    data = pd.read_csv(data_path)

    data["percent.positive"] = data["percent.positive"].fillna(0)
    data["total.specimens"] = data["total.specimens"].fillna(0)
    for column in ["median_t", "median_tmin", "median_tmax", "total_prcp",
                   "humidity"]:

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


def run_all():

    """Runs all eda, with option for models"""
    data = get_data()

    fit_and_plot(data, linear=True, decision=True,
                 bayes=True, test_d=False)


if __name__ == "__main__":
    print("Running")
    run_all()
    print("Done")
