from sklearn import neighbors, linear_model, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
# label 60 total news shares

# Titulo todo, si tiene imagen, video.
# Weekdays, global rate positive and negative

# Importar y procesar los datos

# df = data frame
df = pd.read_csv("OnlineNewsPopularity.csv", sep=", ")

# print(df)

weekdays = df[["weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday", "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday", "weekday_is_sunday"]]


days = weekdays.mul(range(7), fill_value=0)
days = days.agg("sum", axis="columns")
df[["days"]] = days

c = df[["data_channel_is_lifestyle", "data_channel_is_entertainment", "data_channel_is_bus", "data_channel_is_socmed", "data_channel_is_tech", "data_channel_is_world"]]
channels = c.mul(range(6), fill_value=0)
channels = channels.agg("sum", axis="columns")
df[["channels"]] = channels

# X sera un dataframe con solo las columnas descritas
# X = df[["global_rate_positive_words", "global_rate_negative_words", "num_imgs", "num_videos", "days"]]
X = df[["rate_positive_words", "rate_negative_words", "num_imgs", "num_videos", "days",]]
# y será un DataFrame con solo la columna shares
y = df[["shares"]]

#subplots, dentro de una grafica poner varias graficas
column_names = ["rate_positive_words", "rate_negative_words", "num_imgs", "num_videos", "days", "num_hrefs"]

# 6 datos -> 2 rows x 3 columns

fig, axs = plt.subplots(2,3)
for index, ax in enumerate(axs.flat):
    ax.scatter(df[[column_names[index]]], y, s=0.5)
    ax.set_title(column_names[index])
    ax.set(ylabel="scores")

# train_test_split
# test_size => porcentaje (float) de datos de pruebas 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=93)



# model
model = neighbors.KNeighborsRegressor(n_neighbors=5)
# model = linear_model.LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print(y_test)
# print(y_pred)

#Accuracy

acc = model.score(X_test, y_test)

print(f"Accuracy: {acc}")