import pandas as pd
import numpy as np
#
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Activation, Dropout
# from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn import preprocessing, model_selection, neighbors
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans


def del_asterisks(df):
	for column in df.drop(columns=['Tumor type', 'Age', 'Sex', 'Race']):
		myList = df[column].tolist()
		for index, value in enumerate(myList):
			if isinstance(value, str) == True:
				newValue = value.replace('*', '')
				newValue = float(newValue)
				myList[index] = newValue
		df[column] = myList
	return df


def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = list(set(column_contents))
            unique_elements.sort()
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df


def one_hot(array):
    """One hot encoding"""
    unique_list = list(set(array))

    mod_array = []
    for value in array:
        temp_row = np.zeros(len(unique_list))
        temp_row[unique_list.index(value)] = 1
        mod_array.append(temp_row)

    return mod_array

df = pd.read_excel("CancerSEEK01.xlsx")
df_mod = df.drop(columns = ['Sample ID #', 'Prediction', 'label'])

df_mod.fillna(-99999, inplace=True)
df_mod = del_asterisks(df_mod)
df_mod = handle_non_numerical_data(df_mod)

X = np.array(df_mod.drop(columns=['Tumor type']), dtype=float)
X = preprocessing.scale(X)

y = np.array(df_mod['Tumor type'])
y = np.array(one_hot(y))

i=10
input_data = df_mod.drop(columns=["Tumor type"])
input_data.loc[i].to_json("sample_test{}.json".format(i))


print(X.shape)
print(y.shape)


# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
#
# model = Sequential()
# model.add(Dense(20, input_shape=(46,), activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(9, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# checkpointer = ModelCheckpoint('multi-class-model.h5', verbose=1, save_best_only=True)
#
# model.fit(X_train, y_train, validation_split=0.2, epochs=40, batch_size=2, callbacks=[checkpointer])
#
# print(model.evaluate(X_test, y_test))
# model.save("multi-class-model.h5")






X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
