import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.decomposition import TruncatedSVD
import sklearn.linear_model as lin
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from scipy.stats.mstats import zscore

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


def fit_violin(y_test, y_pred, type = "knn", depth=15):

    """Makes a violin plot of the predicted vs test results"""

    fig, axes = plt.subplots(nrows=1, figsize=(14, 9))#, ncols=2)

    axes.set_xlabel("Season")
    axes.set_ylabel("Cases (tens of thousands)")

    fit_data = [y_test, y_pred]

    axes.violinplot(fit_data, showmedians=True)

    plt.title("Predicted 2017 cases (" + type + ")")
    plt.setp(axes, xticks=[1, 2], xticklabels=["Test", "Predicted"])
    axes.yaxis.grid(True, color='w')


    if type == "knn" or type == "decision":

        plt.savefig(save_path + "fit_vs_pred_violin_" + type +
                    str(depth) + "8.pdf")
        print("saved at " + save_path + "fit_vs_pred_violin_" + type +
              str(depth) + "8.pdf")
    else:
        plt.savefig(save_path + "fit_vs_pred_violin_" + type + "8.pdf")
        print("saved at " + save_path + "fit_vs_pred_violin_" + type +
              "8.pdf")
    plt.close()


def fit_and_plot(data, knn=True, linear=True, decision=True,
                 sgd=False, neural=False, ridge=False,
                 random_forest=False, bayes=False, lasso=False, svd=False,
                 ada=False, logistic=False):

    """Fits and plots results for specified models"""

    plt.close()


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

    for column in X_cols:

        if column != 'vac_effectiveness' and column != "vac_num":

            X_train[column] = (X_train[[column]] - X_train[[column]].mean()) / X_train[[column]].std()
            X_test[column] = (X_test[[column]] - X_test[[column]].mean()) / X_test[[column]].std()

    if knn:
        """KNN"""
        # best_rmse = 1e9
        # best_k = -1
        # for k in range(3, 50):
        #     regressor = KNeighborsRegressor(n_neighbors=k)
        #     fit = regressor.fit(X_train, y_train)
        #     y_pred = fit.predict(X_test)
        #     rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        #     rmse = rmse.__round__(4)
        #     print(rmse, k)
        #     if rmse < best_rmse:
        #         best_rmse = rmse
        #         best_k = k
        #
        # print("best k = " + str(best_k))
        # k = best_k
        k = 12
        print("knn")
        regressor = KNeighborsRegressor(n_neighbors=k)
        fit = regressor.fit(X_train, y_train)
        r2 = fit.score(X_train, y_train)
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        print("rmse " + str(rmse) + "k " + str(k))
        r2=r2.__round__(4)
        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 Cases KNN")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1., "k = " + str(k))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path + "knn_" + str(k) +
                    "_predicted_cases_2017_num" + str(k) + "8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="knn", depth=k)


    if linear:
        """Linear"""
        print("Linear")
        regressor = lin.LinearRegression()
        fit = regressor.fit(X_train, y_train)
        coefficients = fit.coef_
        print("coefficients ")
        print(coefficients)
        r2 = fit.score(X_train, y_train)
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        print("rmse " + str(rmse))
        r2=r2.__round__(4)
        # y_pred = y_test.mean() + y_pred*y_test.std()

        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 Cases Linear")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path + "lin_predicted_cases_2017_num8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="linear")

    if logistic:
        """Linear"""
        print("Logistic")
        regressor = lin.LogisticRegression()
        fit = regressor.fit(X_train, y_train)
        # coefficients = fit.coef_
        # print("coefficients ")
        # print(coefficients)
        r2 = fit.score(X_train, y_train)
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        print("rmse " + str(rmse))
        r2=r2.__round__(4)
        # y_pred = y_test.mean() + y_pred*y_test.std()

        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 Cases Logistic")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path + "log_predicted_cases_2017_num8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="log")

    if decision:
        """Decision tree"""
        # best_rmse = 1e9
        # best_depth = -1
        # for depth in range(6, 15):
        #     regressor=tree.DecisionTreeRegressor(max_depth=depth)
        #     fit = regressor.fit(X_train, y_train)
        #     y_pred = fit.predict(X_test)
        #     rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        #     rmse = rmse.__round__(4)
        #     print(rmse, depth)
        #     if rmse < best_rmse:
        #         best_rmse = rmse
        #         best_depth = depth
        #
        # print("best depth = " + str(depth))
        depth = 10
        print("decision")
        regressor = tree.DecisionTreeRegressor(max_depth=depth)
        fit = regressor.fit(X_train, y_train)
        r2 = fit.score(X_train, y_train)
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        print("rmse " + str(rmse))
        r2=r2.__round__(4)
        # y_pred = y_pred.mean() + y_pred*y_pred.std()

        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 Cases Decision Tree")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1., "Depth = " + str(depth))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path +
                    "d_tree_predicted_cases_2017_num" + str(depth) + "8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="decision", depth=depth)

    if random_forest:

        """Random forest"""
        regressor = RandomForestRegressor()
        print("random")
        # label_enc = preprocessing.LabelEncoder()
        # y_train = label_enc.fit_transform(y_train)
        fit = regressor.fit(X_train, y_train)
        r2 = fit.score(X_train, y_train)
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        print("rmse " + str(rmse))
        r2=r2.__round__(4)
        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 Cases Random Forest")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path + "rand_forest_predicted_cases_2017_num8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="random_forest")

    if neural:
        """Neutral net"""
        regressor = MLPRegressor()
        print("neural")
        # label_enc = preprocessing.LabelEncoder()
        # y_train = label_enc.fit_transform(y_train)
        fit = regressor.fit(X_train, y_train)
        r2 = fit.score(X_train, y_train)
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        if min(y_pred) < 0:
            print(min(y_pred))
            y_pred += abs(min(y_pred))
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        print("rmse " + str(rmse))
        r2=r2.__round__(4)
        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 Cases Neural Network")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path + "neural_net_predicted_cases_2017_num8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="neural")

    if sgd:
        """Stochastic gradient descent"""
        regressor=lin.SGDRegressor()
        print("sgd")
        # label_enc = preprocessing.LabelEncoder()
        # y_train = label_enc.fit_transform(y_train)
        fit = regressor.fit(X_train, y_train)
        r2 = fit.score(X_train, y_train)
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        print("rmse " + str(rmse))
        r2=r2.__round__(4)
        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 SGD")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path + "sgd_predicted_cases_2017_num8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="sgd")

    if ridge:
        """Ridge regression"""
        regressor = lin.Ridge()
        print("ridge")
        fit = regressor.fit(X_train, y_train)
        r2 = fit.score(X_train, y_train)
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        print("rmse " + str(rmse))
        r2=r2.__round__(4)
        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 Ridge Regression")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path + "rr_predicted_cases_2017_num8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="ridge")

    if ada:
        """Ridge regression"""
        regressor = AdaBoostRegressor()
        print("ada")
        fit = regressor.fit(X_train, y_train)
        r2 = fit.score(X_train, y_train)
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        print("rmse " + str(rmse))
        r2=r2.__round__(4)
        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 Ada Boost")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path + "ada_predicted_cases_2017_num8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="ada")

    if lasso:
        """Lasso least angle regression"""
        regressor = lin.LassoLars()

        print("lasso")
        fit = regressor.fit(X_train, y_train)
        r2 = fit.score(X_train, y_train)
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        print("rmse " + str(rmse))
        r2=r2.__round__(4)
        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 Lasso Least Angle Regression")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path + "lasso_predicted_cases_2017_num8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="elasso")

    if svd:
        """Ridge regression"""
        regressor = TruncatedSVD()
        print("svd")
        fit = regressor.fit(X_train, y_train)
        r2 = fit.score(X_train, y_train)
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        print("rmse " + str(rmse))
        r2=r2.__round__(4)

        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 Truncated SVD")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path + "svd_predicted_cases_2017_num8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="svd")

    if bayes:
        """Bayesian Ridge regression"""
        regressor = lin.BayesianRidge()

        # label_enc = preprocessing.LabelEncoder()
        # y_train = label_enc.fit_transform(y_train)
        fit = regressor.fit(X_train, y_train)
        r2 = fit.score(X_train, y_train)
        print("bayes")
        print("r^2 = " + str(r2))
        y_pred = fit.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse = rmse.__round__(4)
        r2 = r2.__round__(4)
        print("rmse " + str(rmse))

        plt.scatter([i for i in range(len(y_test))], y_test, s=3,
                    label="Actual")
        plt.scatter([i for i in range(len(y_test))], y_pred, s=3,
                    label="Predicted")
        # plt.xticks(tick_locs, x_ticks)
        plt.legend(loc="best")
        plt.xlabel("Season week")
        plt.ylabel("Cases (tens of thousands)")
        plt.title("Predicted 2017 Bayesian Ridge Regression")
        plt.text(40, 1.3, "RMSE = " + str(rmse))
        plt.text(40, 1.5, r"$R^{2}$ = " + str(r2))
        # plt.show()
        plt.savefig(save_path + "brr_predicted_cases_2017_num8.pdf")
        plt.close()
        fit_violin(y_test, y_pred, type="bayes")

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


def run_all():

    """Runs all eda, with option for models"""
    data = get_data()

    fit_and_plot(data, linear=True, knn=True, decision=True,
                 sgd=True, neural=True, ridge=True,
                 random_forest=True, bayes=True, lasso=True,
                 ada=True)


if __name__ == "__main__":
    print("Running")
    run_all()
    print("Done")