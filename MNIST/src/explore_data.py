import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection


if __name__ == '__main__':
    df = pd.read_csv('F:\Work\Project Portfolio\PROJECT - ML Minis\MNIST\input\mnist_train.csv')
    print(df.head())

    b = sns.distplot(x='label', data=df)
    b.set_xlabel('label', fontsize=20)
    b.set_ylabel('count', fontsize=20)
    plt.show()
