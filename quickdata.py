import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class data_viz:

    def multicollinearity_check(data, target, remove=[], add_target=False, inplace=False):
        """This function allows the easy visualisation of multicollinearity between variables
        input data is of the following format 'data' = X , target = y, remove """
        f, ax = plt.subplots(figsize=(12, 10))
        df = pd.DataFrame()

        if add_target == True:
            data["target"] = target

        if len(remove) != 0 and remove[0] != "":
            for item in remove:
                if item in list(data.columns):
                    df[item] = data[item]

            data_dropped = data.drop(remove, axis=1)
        else:
            data_dropped = data

        corr_matrix = data_dropped.corr()
        mask = np.zeros_like(corr_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        heatmap = sns.heatmap(round(corr_matrix, 2), mask=mask, square=True,
                              linewidths=.5, cmap="YlGnBu",
                              cbar_kws={"shrink": .4,
                                        "ticks": [-1, -.5, 0, 0.5, 1]},
                              vmin=-1, vmax=1, annot=True,
                              annot_kws={"size": 14, "color": "black", "backgroundcolor": "white", "bbox": dict(facecolor='white', alpha=0.5)})
        # add the column names as labels

        Title_font = {
            'color':  'Black',
            'weight': 'normal',
            'size': 20,
        }
        if add_target != True:
            plt.title(
                "Multicollinearity between Independent Variables\n", fontdict=Title_font)
        else:
            plt.title(
                "Multicollinearity between Independent and Target Variables\n", fontdict=Title_font)

        ax.set_yticklabels(corr_matrix.columns, rotation=0)
        ax.set_xticklabels(corr_matrix.columns)
        ax.xaxis.set_ticks_position("top")
        relation = dict()

        for column in list(corr_matrix.columns):
            for index in corr_matrix.index:
                value = corr_matrix[column][index]
                if (abs(value) > 0.7) and (abs(value) != 1) and (column != "target" or index != "target"):
                    if column not in relation.values():
                        relation.update({column: index})
        for key, value in relation.items():
            print("-------------------------------------------")
            print("Warning!! High collinearity found between variables {} and {}".format(
                key, value))

        if len(remove) == 1 and remove[0] != "target":
            fig = plt.figure(figsize=(12, 4))
            p = sns.regplot(x=data[remove[0]], y=target)
        elif len(remove) >= 1 and remove[0] != "target":
            fig, axs = plt.subplots(len(remove), 1, figsize=(12, 8))
            for i, index in enumerate(remove):
                p = sns.regplot(x=data[index], y=target, ax=axs[i])

        print("-------------------------------------------")

        if inplace == True:
            if add_target == False:
                if "target" in list(data.columns):
                    data.drop(["target"], axis=1, inplace=True)
            if len(remove) > 0:
                for column in list(data.columns):
                    if column in list(df.columns):
                        print(column)

                        data.drop(column, axis=1, inplace=True)

        elif inplace == False:
            if add_target == True:
                if "target" in list(data.columns):
                    data.drop(["target"], axis=1, inplace=True)
            for column in remove:
                if column not in data.columns:
                    data[column] = df[column]

        X = data

    def view_outliers(data, target, split_size=0.3):
        """This function returns a side-by-side view of outliers through a regplot 
        and a boxplot visualisationin of a dataframe'data' of independent variables
        as columns within that dataframe, other parameters include 'target' the 
        dependent variable and 'split_size' to adjust the number of plotted rows 
        as decimals between 0 and 1 or as integers"""

        if type(split_size) == float:
            if split_size % 1 != 0:
                split_size = int(len(data)*split_size)
        elif type(split_size) == str:
            if split_size == "no":
                split_size = int(len(data))
        fig, axs = plt.subplots(len(data.columns) - 2, 2, figsize=(30, 40))
        for index in list(range(0, len(data.columns)-2)):
            p = sns.regplot(x=data[list(data.columns)[
                            index]][:split_size], y=target[:split_size], ax=axs[index][0])
        #         axs[index][0].plot(p.get_lines()[0].get_xdata(),p.get_lines()[0].get_ydata(),c='green')

            sns.boxplot(data[list(data.columns)[index]]
                        [:split_size], ax=axs[index][1])

    def view_missing(data):
        sns.heatmap(data.isnull(), cbar=False)
