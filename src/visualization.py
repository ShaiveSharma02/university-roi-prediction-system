import matplotlib.pyplot as plt


def plot_feature_importance(model, feature_columns):
    importance_values = model.feature_importances_

    fig, ax = plt.subplots()

    ax.barh(feature_columns, importance_values)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    return fig