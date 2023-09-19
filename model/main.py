import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle as pk

def createModel(palmerPenguins):

    df = palmerPenguins.copy()
    target = 'species'
    encode = ['sex', 'island']

    # encode the input variable 
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df,dummy], axis=1)
        del df[col]

    # encode target variable
    target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
    def target_encode(val):
        return target_mapper[val]
    
    df['species'] = df['species'].apply(target_encode)
    print(df.head())
    
    x = df.drop(['species'], axis=1)
    y = df['species']

    # split the data
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, 
                                                    test_size=0.2, random_state=42)
    # train
    model = RandomForestClassifier()
    model.fit(xTrain, yTrain)

    # test the model
    yPred = model.predict(xTest)

    print("Accuracy:", accuracy_score(yTest, yPred))
    print("Classification Report: \n", classification_report(yTest, yPred))

    return model



def main():

    # Dataset Link
    # https://github.com/dataprofessor/code/blob/master/streamlit/part3/penguins_cleaned.csv

    palmerPenguins = pd.read_csv("../data/penguins_cleaned.csv")

    # print(palmerPenguins.describe())

    model = createModel(palmerPenguins)

    # with open('../model/model.pkl', 'wb') as f:
    #     pk.dump(model, f)
        
if __name__ == '__main__':
    main()