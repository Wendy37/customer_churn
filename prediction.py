import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def logistic_regression(df, y, headers):
    scaler = MinMaxScaler()
    df[headers] = scaler.fit_transform(df[headers])
    x_train, x_test, y_train, y_test = train_test_split(df, y, stratify=y)
    logist_result(y, df)
    logreg = LogisticRegression(multi_class='multinomial')
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    ROC_curve(logreg, x_test, y_test)


def logist_result(y_test, x_test):  # use for logistic regression, summary of results
    import statsmodels.api as sm
    logit_model = sm.Logit(y_test, x_test).fit()
    print(logit_model.summary())


def ROC_curve(logreg, x_test, y_test):
    # calculate the fpr and tpr for all thresholds of the classification
    probs = logreg.predict_proba(x_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def tree(df, y, depth):
    x_train, x_test, y_train, y_test = train_test_split(df, y, stratify=y)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    feature_importance(clf)


def feature_importance(clf):  # this method show feature importance for decision tree model
    fi = clf.feature_importances_
    for i, v in enumerate(fi):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    plt.bar([x for x in range(len(fi))], fi)
    plt.show()


def check_imbalance(df):
    g = sns.countplot(df['LEAVE'])
    g.set_xticklabels(['stay', 'leave'])
    plt.show()


def plot_fitting_curve(datasets, maxdepth=15):
    # Intialize accuracies
    accuracies = {}
    for key in datasets:
        accuracies[key] = []
    # Initialize depths
    depths = range(1, maxdepth+1)
    # Fit model for each specific depth
    for md in depths:
        model = DecisionTreeClassifier(max_depth=md, random_state=42)
        # Record accuracies
        for key in datasets:
            X = datasets[key]['X']
            Y = datasets[key]['Y']
            if key == "X-Val":
                accuracies[key].append(cross_val_score(model, X, Y, scoring="accuracy", cv=5).mean())
            else:
                model.fit(datasets['Train']['X'], datasets['Train']['Y'])
                accuracies[key].append(accuracy_score(model.predict(X), Y))
    # Plot each curve
    plt.figure(figsize=[10,7])
    for key in datasets:
        plt.plot(depths, accuracies[key], label=key)
    # Plot details
    plt.title("Performance on train and test data")
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy")
    # Find minimum accuracy in all runs
    min_acc = np.array(list(accuracies.values())).min()
    plt.ylim([min_acc, 1.0])
    plt.xlim([1, maxdepth])
    plt.legend()
    plt.grid()
    plt.show()


def k_fold_cross_valid(df, y, logist=True, tree=False):
    from sklearn.model_selection import cross_val_score
    if logist:
        model = LogisticRegression()
    else:
        model = DecisionTreeClassifier(max_depth=10)
    scores = cross_val_score(model, df, y, scoring="accuracy", cv=10)
    print("Cross Validated Accuracy: %0.3f +/- %0.3f" % (scores.mean(), scores.std()))


def tree_fitting(df, y):
    X_train, X_test, Y_train, Y_test = train_test_split(df, y, test_size=0.20)
    datasets = {"Train": {"X": X_train, "Y": Y_train},
                "Test": {"X": X_test, "Y": Y_test},
                "X-Val": {"X": df, "Y": y}}
    plot_fitting_curve(datasets, maxdepth=10)


if __name__ == '__main__':
    df = pd.read_csv("Customer_Churn.csv")
    df.dropna(axis=0)  # handling missing data, case-wise deletion
    y = df.iloc[:, -1]
    y = pd.get_dummies(y)
    y = y.drop('STAY', axis=1)  # y
    df = df.drop('LEAVE', axis=1)
    df = pd.get_dummies(df)  # x
    headers = [col for col in df.columns]
    logistic_regression(df, y, headers)
    #tree(df, y, 3)



