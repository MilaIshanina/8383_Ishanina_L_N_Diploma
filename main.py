
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree


heart_df = pd.read_csv("heart.csv")

heart_df['Cholesterol'].replace(0, heart_df['Cholesterol'].median(), inplace=True)
heart_df = pd.get_dummies(heart_df, drop_first=True)

heart_df_copy = heart_df.copy()
# x_train = heart_df_copy.drop('HeartDisease',axis=1)
# y_train = heart_df_copy.pop('HeartDisease')
x = heart_df.drop(["HeartDisease"], axis=1)
y = heart_df["HeartDisease"]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1)

all_cols = [cname for cname in x_train.columns]
# print(all_cols)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import plot_partial_dependence


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def corr_plot():
    corr = heart_df.corr().round(2)
    plt.figure(figsize=(13, 13))

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    heat_map = sns.heatmap(corr,
                           annot=True,

                           linewidths=0.01,
                           cbar_kws={'shrink': .5})
    plt.show()
    return 0


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# MODELS:
#  KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)

#  DecisionTreeClassifier
dt = tree.DecisionTreeClassifier(max_depth=5)
dt.fit(x_train, y_train)
dt.predict(x_test)

#  LogisticRegression

from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(random_state=1)
logit.fit(x_train, y_train)
pred_logit = logit.predict(x_test)

#  SVC

from sklearn.svm import SVC

svc_clf = SVC(kernel='rbf', probability=True)
svc_clf.fit(x_train, y_train)
pred = svc_clf.predict(x_test)

#  GradientBoostingClassifier


gb_clf = GradientBoostingClassifier(n_estimators=100).fit(x_train, y_train)
pred_gb = gb_clf.predict(x_test)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ACCURACY

def ac():
    print("Accuracy KNN:")
    print(knn.score(x_test, y_test))

    print("Accuracy DecisionTreeClassifier:")
    print(dt.score(x_test, y_test))

    print("Accuracy LogisticRegression:")
    print(logit.score(x_test, y_test))

    print("Accuracy GradientBoostingClassifier:")
    print(gb_clf.score(x_test, y_test))

    print("Accuracy SVC:")
    print(svc_clf.score(x_test, y_test))

    return 0


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# PDP
import time


def pdp_p(name):
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=(15, 32))
    if name == 'GradientBoostingClassifier':
        plt.figtext(0.39, 0.9, 'Partial Dependence Plot, PDP\n GradientBoostingClassifier', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(gb_clf, x_test, all_cols, ax=ax)
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'KNeighborsClassifier':
        plt.figtext(0.39, 0.9, 'Partial Dependence Plot, PDP\n KNeighborsClassifier', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(knn, x_test, all_cols, ax=ax)
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'DtTree':
        plt.figtext(0.39, 0.9, 'Partial Dependence Plot, PDP\n DecisionTreeClassifier', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(dt, x_test, all_cols, ax=ax)
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'LogisticRegression':
        plt.figtext(0.39, 0.9, 'Partial Dependence Plot, PDP\n LogisticRegression', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(logit, x_test, all_cols, ax=ax)
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'SVC':
        plt.figtext(0.39, 0.9, 'Partial Dependence Plot, PDP\n SVC', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(svc_clf, x_test, all_cols, ax=ax)
        print("--- %s seconds ---" % (time.time() - start_time))

    print("\n")
    return 0


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ICE


def ice_p(name):
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=(15, 32))
    if name == 'GradientBoostingClassifier':
        plt.figtext(0.35, 0.9, 'Individual Conditional Expectation Plot, ICE', fontsize=18)
        plt.figtext(0.40, 0.89, 'GradientBoostingClassifier', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(gb_clf, x_test, all_cols, ax=ax, kind="individual")
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'KNeighborsClassifier':
        plt.figtext(0.35, 0.9, 'Individual Conditional Expectation Plot, ICE', fontsize=18)
        plt.figtext(0.40, 0.89, 'KNeighborsClassifier', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(knn, x_test, all_cols, ax=ax, kind="individual")
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'DtTree':
        plt.figtext(0.35, 0.9, 'Individual Conditional Expectation Plot, ICE', fontsize=18)
        plt.figtext(0.40, 0.89, 'DecisionTreeClassifier', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(dt, x_test, all_cols, ax=ax, kind="individual")
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'LogisticRegression':
        plt.figtext(0.35, 0.9, 'Individual Conditional Expectation Plot, ICE', fontsize=18)
        plt.figtext(0.40, 0.89, 'LogisticRegression', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(logit, x_test, all_cols, ax=ax, kind="individual")
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'SVC':
        plt.figtext(0.35, 0.9, 'Individual Conditional Expectation Plot, ICE', fontsize=18)
        plt.figtext(0.40, 0.89, 'SVC', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(svc_clf, x_test, all_cols, ax=ax, kind="individual")
        print("--- %s seconds ---" % (time.time() - start_time))

    print("\n")
    return 0;


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def ice_and_pdp(name):
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=(15, 32))
    if name == 'GradientBoostingClassifier':
        plt.figtext(0.35, 0.9, 'Individual Conditional Expectation Plot, ICE', fontsize=18)
        plt.figtext(0.40, 0.89, 'GradientBoostingClassifier', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(gb_clf, x_test, all_cols, ax=ax, kind="both")
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'KNeighborsClassifier':
        plt.figtext(0.35, 0.9, 'Individual Conditional Expectation Plot, ICE', fontsize=18)
        plt.figtext(0.40, 0.89, 'KNeighborsClassifier', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(knn, x_test, all_cols, ax=ax, kind="both")
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'DtTree':
        plt.figtext(0.35, 0.9, 'Individual Conditional Expectation Plot, ICE', fontsize=18)
        plt.figtext(0.40, 0.89, 'DecisionTreeClassifier', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(dt, x_test, all_cols, ax=ax, kind="both")
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'LogisticRegression':
        plt.figtext(0.35, 0.9, 'Individual Conditional Expectation Plot, ICE', fontsize=18)
        plt.figtext(0.40, 0.89, 'LogisticRegression', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(logit, x_test, all_cols, ax=ax, kind="both")
        print("--- %s seconds ---" % (time.time() - start_time))
    elif name == 'SVC':
        plt.figtext(0.35, 0.9, 'Individual Conditional Expectation Plot, ICE', fontsize=18)
        plt.figtext(0.40, 0.89, 'SVC', fontsize=18)
        start_time = time.time()
        plot_partial_dependence(svc_clf, x_test, all_cols, ax=ax, kind="both")
        print("--- %s seconds ---" % (time.time() - start_time))

    print("\n")
    return 0

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




import matplotlib as mpl
from alepython import ale_plot
from alibi.explainers import ALE, plot_ale


def ale_p(name):
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=(15, 32))
    if name == 'GradientBoostingClassifier':
        start_time = time.time()
        ale = ALE(gb_clf.predict, feature_names=all_cols, target_names=['HeartDisease'])
        exp = ale.explain(x_test.values)
        plot_ale(exp, fig_kw={'figwidth': 14, 'figheight': 27})
        print("--- %s seconds ---" % (time.time() - start_time))

    elif name == 'KNeighborsClassifier':
        start_time = time.time()
        ale = ALE(knn.predict, feature_names=all_cols, target_names=['HeartDisease'])
        exp = ale.explain(x_test.values)
        plot_ale(exp, fig_kw={'figwidth': 14, 'figheight': 27})
        print("--- %s seconds ---" % (time.time() - start_time))

    elif name == 'DtTree':
        start_time = time.time()
        ale = ALE(dt.predict, feature_names=all_cols, target_names=['HeartDisease'])
        exp = ale.explain(x_test.values)
        plot_ale(exp, fig_kw={'figwidth': 14, 'figheight': 27})
        print("--- %s seconds ---" % (time.time() - start_time))

    elif name == 'LogisticRegression':
        start_time = time.time()
        ale = ALE(logit.predict, feature_names=all_cols, target_names=['HeartDisease'])
        exp = ale.explain(x_test.values)
        plot_ale(exp, fig_kw={'figwidth': 14, 'figheight': 27})
        print("--- %s seconds ---" % (time.time() - start_time))

    elif name == 'SVC':
        start_time = time.time()
        ale = ALE(svc_clf.predict, feature_names=all_cols, target_names=['HeartDisease'])
        exp = ale.explain(x_test.values)
        plot_ale(exp, fig_kw={'figwidth': 14, 'figheight': 27})
        print("--- %s seconds ---" % (time.time() - start_time))

    print("\n")
    return 0;

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# SHAP

import shap


def shap_p(name):
    if name == 'GradientBoostingClassifier':
        explainer = shap.Explainer(gb_clf)
        shap_values = explainer.shap_values(x_test)
        start_time = time.time()
        shap.summary_plot(shap_values, x_test, max_display=25, auto_size_plot=True)
        print("--- %s seconds ---" % (time.time() - start_time))

    elif name == 'KNeighborsClassifier':
        explainer = shap.Explainer(knn)
        shap_values = explainer.shap_values(x_test)
        start_time = time.time()
        shap.summary_plot(shap_values, x_test, max_display=25, auto_size_plot=True)
        print("--- %s seconds ---" % (time.time() - start_time))

    elif name == 'DtTree':
        explainer = shap.Explainer(dt)
        shap_values = explainer.shap_values(x_test)
        start_time = time.time()
        shap.summary_plot(shap_values, x_test, max_display=25, auto_size_plot=True)
        print("--- %s seconds ---" % (time.time() - start_time))

    elif name == 'LogisticRegression':
        explainer = shap.Explainer(logit)
        shap_values = explainer.shap_values(x_test)
        start_time = time.time()
        shap.summary_plot(shap_values, x_test, max_display=25, auto_size_plot=True)
        print("--- %s seconds ---" % (time.time() - start_time))

    elif name == 'SVC':
        explainer = shap.Explainer(svc_clf)
        shap_values = explainer.shap_values(x_test)
        start_time = time.time()
        shap.summary_plot(shap_values, x_test, max_display=25, auto_size_plot=True)
        print("--- %s seconds ---" % (time.time() - start_time))

    print("\n")
    return 0;


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.inspection import permutation_importance


def f_i(name):
    if name == 'GradientBoostingClassifier':
        start_time = time.time()
        result = permutation_importance(gb_clf, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        importances = pd.Series(result.importances_mean, index=all_cols)
        plt.rcParams.update({'font.size': 13})
        fig, ax = plt.subplots(figsize=(10, 8))
        importances.sort_values().plot.barh(yerr=result.importances_std, ax=ax)
        print("--- %s seconds ---" % (time.time() - start_time))
        ax.set_title("Permutation Feature Importance")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.tight_layout()
        plt.show()

    elif name == 'KNeighborsClassifier':
        start_time = time.time()
        result = permutation_importance(knn, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        importances = pd.Series(result.importances_mean, index=all_cols)
        plt.rcParams.update({'font.size': 13})
        fig, ax = plt.subplots(figsize=(10, 8))
        importances.sort_values().plot.barh(yerr=result.importances_std, ax=ax)
        print("--- %s seconds ---" % (time.time() - start_time))
        ax.set_title("Permutation Feature Importance")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.tight_layout()
        plt.show()

    elif name == 'DtTree':
        start_time = time.time()
        result = permutation_importance(dt, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        importances = pd.Series(result.importances_mean, index=all_cols)
        plt.rcParams.update({'font.size': 13})
        fig, ax = plt.subplots(figsize=(10, 8))
        importances.sort_values().plot.barh(yerr=result.importances_std, ax=ax)
        print("--- %s seconds ---" % (time.time() - start_time))
        ax.set_title("Permutation Feature Importance")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.tight_layout()
        plt.show()

    elif name == 'LogisticRegression':
        start_time = time.time()
        result = permutation_importance(logit, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        importances = pd.Series(result.importances_mean, index=all_cols)
        plt.rcParams.update({'font.size': 13})
        fig, ax = plt.subplots(figsize=(10, 8))
        importances.sort_values().plot.barh(yerr=result.importances_std, ax=ax)
        print("--- %s seconds ---" % (time.time() - start_time))
        ax.set_title("Permutation Feature Importance")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.tight_layout()
        plt.show()

    elif name == 'SVC':
        start_time = time.time()
        result = permutation_importance(svc_clf, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        importances = pd.Series(result.importances_mean, index=all_cols)
        plt.rcParams.update({'font.size': 13})
        fig, ax = plt.subplots(figsize=(10, 8))
        importances.sort_values().plot.barh(yerr=result.importances_std, ax=ax)
        print("--- %s seconds ---" % (time.time() - start_time))
        ax.set_title("Permutation Feature Importance")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.tight_layout()
        plt.show()

    print("\n")
    return 0;


print("Hello! Please select an action: \n")

while True:
    print(
        "0 - Correlation matrix \n1 - Accuracy of all models \n2 - PDP plots \n3 - ICE plots \n4 - PDP&ICE plots \n5 - ALE plots \n6 - SHAP plot \n7 - PFI plot \n8 - Exit")
    act = int(input())

    if act == 0:
        corr_plot()
        print("\n")
    elif act == 1:
        ac()
        print("\n")

    elif act == 2:
        print("\n")
        print("Please choose a model:\n")
        print(
            "0 - KNN \n1 - DecisionTreeClassifier \n2 - LogisticRegression \n3 - GradientBoostingClassifier \n4 - SVC \n5 - Exit")
        m = int(input())
        if m == 0:
            pdp_p('KNeighborsClassifier')
            print("\n")
        elif m == 1:
            pdp_p('DtTree')
            print("\n")
        elif m == 2:
            pdp_p('LogisticRegression')
            print("\n")
        elif m == 3:
            pdp_p('GradientBoostingClassifier')
            print("\n")
        elif m == 4:
            pdp_p('SVC')
            print("\n")
        elif m == 5:
            print("\n")

    elif act == 3:
        print("\n")
        print("Please choose a model:\n")
        print(
            "0 - KNN \n1 - DecisionTreeClassifier \n2 - LogisticRegression \n3 - GradientBoostingClassifier \n4 - SVC \n5 - Exit")
        m = int(input())
        if m == 0:
            ice_p('KNeighborsClassifier')
            print("\n")
        elif m == 1:
            ice_p('DtTree')
            print("\n")
        elif m == 2:
            ice_p('LogisticRegression')
            print("\n")
        elif m == 3:
            ice_p('GradientBoostingClassifier')
            print("\n")
        elif m == 4:
            ice_p('SVC')
            print("\n")
        elif m == 5:
            print("\n")

    elif act == 4:
        print("\n")
        print("Please choose a model:\n")
        print(
            "0 - KNN \n1 - DecisionTreeClassifier \n2 - LogisticRegression \n3 - GradientBoostingClassifier \n4 - SVC \n5 - Exit")
        m = int(input())
        if m == 0:
            ice_and_pdp('KNeighborsClassifier')
            print("\n")
        elif m == 1:
            ice_and_pdp('DtTree')
            print("\n")
        elif m == 2:
            ice_and_pdp('LogisticRegression')
            print("\n")
        elif m == 3:
            ice_and_pdp('GradientBoostingClassifier')
            print("\n")
        elif m == 4:
            ice_and_pdp('SVC')
            print("\n")
        elif m == 5:
            print("\n")

    elif act == 5:
        print("\n")
        print("Please choose a model:\n")
        print(
            "0 - KNN \n1 - DecisionTreeClassifier \n2 - LogisticRegression \n3 - GradientBoostingClassifier \n4 - SVC \n5 - Exit")
        m = int(input())
        if m == 0:
            ale_p('KNeighborsClassifier')
            print("\n")
        elif m == 1:
            ale_p('DtTree')
            print("\n")
        elif m == 2:
            ale_p('LogisticRegression')
            print("\n")
        elif m == 3:
            ale_p('GradientBoostingClassifier')
            print("\n")
        elif m == 4:
            ale_p('SVC')
            print("\n")
        elif m == 5:
            print("\n")

    elif act == 6:
        print("\n")
        print("Please choose a model:\n")
        print(
            "0 - KNN \n1 - DecisionTreeClassifier \n2 - LogisticRegression \n3 - GradientBoostingClassifier \n4 - SVC \n5 - Exit")
        m = int(input())
        if m == 0:
            shap_p('KNeighborsClassifier')
            print("\n")
        elif m == 1:
            shap_p('DtTree')
            print("\n")
        elif m == 2:
            shap_p('LogisticRegression')
            print("\n")
        elif m == 3:
            shap_p('GradientBoostingClassifier')
            print("\n")
        elif m == 4:
            shap_p('SVC')
            print("\n")
        elif m == 5:
            print("\n")

    elif act == 7:
        print("\n")
        print("Please choose a model:\n")
        print(
            "0 - KNN \n1 - DecisionTreeClassifier \n2 - LogisticRegression \n3 - GradientBoostingClassifier \n4 - SVC \n5 - Exit")
        m = int(input())
        if m == 0:
            f_i('KNeighborsClassifier')
            print("\n")
        elif m == 1:
            f_i('DtTree')
            print("\n")
        elif m == 2:
            f_i('LogisticRegression')
            print("\n")
        elif m == 3:
            f_i('GradientBoostingClassifier')
            print("\n")
        elif m == 4:
            f_i('SVC')
            print("\n")
        elif m == 5:
            print("\n")

    elif act == 8:
        print("Goodbye!\n")
        break




