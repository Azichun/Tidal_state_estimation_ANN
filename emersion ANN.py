from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib, re, os, math, pickle, keras
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import seaborn as sns
from nlsfunc import *
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.neighbors import KernelDensity
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import History, EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from suntime import Sun, SunTimeException
from eli5.sklearn import PermutationImportance
from sklearn.decomposition import PCA

np.random.seed(321)
tf.random.set_seed(321)

# create functions to organize data set for LSTM RNN training

def create_LSTM_dataset(data_list, X_cols, y_col, time_step, include_current_data=False):
    X, y = [], []
    for data in data_list:
        try:
            if include_current_data:
                for r in range(data.shape[0] - time_step + 1):
                    X.append(data.iloc[r:r + time_step][X_cols].to_numpy())
                    y.append(data.iloc[r + time_step - 1][y_col].to_numpy())
            else:
                for r in range(data.shape[0] - time_step):
                    X.append(data.iloc[r:r + time_step][X_cols].to_numpy())
                    y.append(data.iloc[r + time_step][y_col].to_numpy())
        except IndexError:
            pass
    if len(y_col) == 0:
        y = None
        return np.array(X), None
    else:
        y = [0 if i == "E" else 1 if i == "A" else 2 for i in y]
        return np.array(X), np.array(to_categorical(y))


#######################
##### Import data #####
#######################

# video

os.chdir("F:\\E_backup\\Mirror\\mphil_videos\\emersion_submersion")
for csv in [csv for csv in os.listdir() if "csv" in csv]:
    dummy = pd.read_csv(csv)
    dummy["Time"] = [datetime.strptime(t, "%m/%d/%y %H:%M") for t in dummy["Time"]]
    dummy["Time"] = dummy["Time"].dt.tz_localize("Hongkong")

    sun = Sun(22.21, 114.26)
    date_list = list(set(dummy["Time"].dt.date))
    date_list.sort()
    for d in date_list:
        sunrise = sun.get_local_sunrise_time(d - timedelta(days=1))
        sunset = sun.get_local_sunset_time(d)
        dummy.loc[dummy["Time"].dt.date == d, "DayNight"] = \
            [sunrise < t < sunset for t in dummy.loc[dummy["Time"].dt.date == d, "Time"]]
    globals()[f"df_{csv[:-4]}"] = dummy


# CFS

os.chdir("F:\\mphil\\Data\\Modelling\\climate_data")
var_list = ["dlwsfc", "dswsfc", "tmp2m", "tcdcclm", "wnd10m"]
month_list = ["202006", "202007", "202008", "202009"]
for var in var_list:
    for month in month_list:
        if var != "wnd10m":
            globals()[f"{var}_{month}"] = pd.read_csv(f"{var}.gdas.{month}.csv", header=None)[[0,1,6]]
        else:
            globals()[f"{var}_{month}"] = pd.read_csv(f"{var}.gdas.{month}.csv", header=None)[[0,1,2,6]]

        drop_index = np.where(globals()[f"{var}_{month}"][0] == globals()[f"{var}_{month}"][1])[0]
        globals()[f"{var}_{month}"].drop(drop_index, inplace=True)
        globals()[f"{var}_{month}"].drop([0], axis=1, inplace=True)
        globals()[f"{var}_{month}"][1] = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in globals()[f"{var}_{month}"][1]]
        globals()[f"{var}_{month}"][1] = [t.tz_convert("Hongkong") for t in globals()[f"{var}_{month}"][1].dt.tz_localize("UTC")]

        if var != "wnd10m":
            globals()[f"{var}_{month}"].columns = ["Time", var]
            try:
                globals()[var] = globals()[var].append(globals()[f"{var}_{month}"])
            except KeyError:
                globals()[var] = globals()[f"{var}_{month}"]
        else:
            u = globals()[f"{var}_{month}"].loc[globals()[f"{var}_{month}"][2] == "UGRD"][6]
            v = globals()[f"{var}_{month}"].loc[globals()[f"{var}_{month}"][2] == "VGRD"][6]
            try: 
                globals()["wind"] = globals()["wind"].append(pd.DataFrame({"Time": globals()[f"{var}_{month}"][1].iloc[::2],
                                                       "wind": np.sqrt(np.array(u) ** 2 + np.array(v) ** 2)}))
                globals()["wdir"] = globals()["wdir"].append(pd.DataFrame({"Time": globals()[f"{var}_{month}"][1].iloc[::2],
                                                       "wdir": [57.29578 * math.atan2(ui, vi) + 180
                                                                for ui, vi in zip(u, v)]}))
            except KeyError:
                globals()["wind"] = pd.DataFrame({"Time": globals()[f"{var}_{month}"][1].iloc[::2],
                                                  "wind": np.sqrt(np.array(u) ** 2 + np.array(v) ** 2)})
                globals()["wdir"] = pd.DataFrame({"Time": globals()[f"{var}_{month}"][1].iloc[::2],
                                                  "wdir": [57.29578 * math.atan2(ui, vi) + 180 for ui, vi in zip(u, v)]})

# SST

os.chdir("F:\\mphil\\Data\\Modelling\\climate_data\\hko")
sstam = pd.read_csv("SSTam_2020.csv")
sstpm = pd.read_csv("SSTpm_2020.csv")

sstam.rename({"Unnamed: 0": "Date"}, axis=1, inplace=True)
sstpm.rename({"Unnamed: 0": "Date"}, axis=1, inplace=True)

sstam = sstam.melt(id_vars=["Date"], var_name="Month", value_name="sst").dropna()
sstpm = sstpm.melt(id_vars=["Date"], var_name="Month", value_name="sst").dropna()

sstam["Full_time"] = [datetime.strptime(f"{d:02}-{m}-2020 06:00", "%d-%b-%Y %H:%M")
                      for d, m in zip(sstam["Date"], sstam["Month"])]
sstpm["Full_time"] = [datetime.strptime(f"{d:02}-{m}-2020 18:00", "%d-%b-%Y %H:%M")
                      for d, m in zip(sstpm["Date"], sstpm["Month"])]

sst = sstam.append(sstpm).sort_values("Full_time")[["Full_time", "sst"]].reset_index(drop=True).\
    rename({"Full_time": "Time"}, axis=1)
sst["Time"] = sst["Time"].dt.tz_localize("Hongkong")

# tide

os.chdir("F:\\mphil\\Tide table")
tide_month_list = {"Jun": "202006", "Jul": "202007", "Aug": "202008", "Sept": "202009"}
tide_list = []
for o_month, n_month in tide_month_list.items():
    dummy = pd.read_excel("Waglan Island 2020.xlsx", engine="openpyxl", sheet_name=o_month,
                                               header=4, index_col=None)
    dummy.rename({"Unnamed: 0": "Date"}, axis=1, inplace=True)
    dummy = pd.melt(dummy, id_vars=["Date"], var_name="Time", value_name="tide")
    dummy.loc[dummy["Time"] == "24:00", "Date"] = dummy.loc[dummy["Time"] == "24:00", "Date"] + 1
    dummy.loc[dummy["Time"] == "24:00", "Time"] = "0:00"
    dummy["Full_time"] = [f"{n_month}{d:02} {t.zfill(5)}" for d, t in zip(dummy["Date"], dummy["Time"])]
    full_time = []
    for t in dummy["Full_time"]:
        try: full_time.append(datetime.strptime(t, "%Y%m%d %H:%M"))
        except ValueError: full_time.append(None)
    dummy["Full_time"] = full_time
    dummy.loc[pd.isnull(dummy["Full_time"]), "Full_time"] = max(dummy["Full_time"]) + timedelta(hours=1)
    tide_list.append(dummy)

for tide in tide_list:
    try:
        tide_total = tide_total.append(tide)
    except NameError:
        tide_total = tide
tide = tide_total.sort_values("Full_time")[["Full_time", "tide"]].reset_index(drop=True).\
    rename({"Full_time": "Time"}, axis=1)
tide["Time"] = tide["Time"].dt.tz_localize("Hongkong")

########################
##### Combine data #####
########################

data_list = [globals()[obj_name] for obj_name in dir() if "df" in obj_name]

for data_frame in data_list:
    data_frame["Temp_diff"] = data_frame["Temp"].diff()
    data_frame.drop(0, inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    data_frame.loc[data_frame["Time"].isin(dswsfc["Time"]), "dswsfc"] = \
        dswsfc.loc[dswsfc["Time"].isin(data_frame["Time"]), "dswsfc"].tolist()
    data_frame.loc[data_frame["Time"].isin(dlwsfc["Time"]), "dlwsfc"] = \
        dlwsfc.loc[dlwsfc["Time"].isin(data_frame["Time"]), "dlwsfc"].tolist()
    data_frame.loc[data_frame["Time"].isin(tcdcclm["Time"]), "tcdcclm"] = \
        tcdcclm.loc[tcdcclm["Time"].isin(data_frame["Time"]), "tcdcclm"].tolist()
    data_frame.loc[data_frame["Time"].isin(tmp2m["Time"]), "tmp2m"] = \
        tmp2m.loc[tmp2m["Time"].isin(data_frame["Time"]), "tmp2m"].tolist()
    data_frame.loc[data_frame["Time"].isin(wind["Time"]), "wind"] = \
        wind.loc[wind["Time"].isin(data_frame["Time"]), "wind"].tolist()
    data_frame.loc[data_frame["Time"].isin(wdir["Time"]), "wdir"] = \
        wdir.loc[wdir["Time"].isin(data_frame["Time"]), "wdir"].tolist()
    data_frame.loc[data_frame["Time"].isin(sst["Time"]), "sst"] = \
        sst.loc[sst["Time"].isin(data_frame["Time"]), "sst"].tolist()
    data_frame.loc[data_frame["Time"].isin(tide["Time"]), "tide"] = \
        tide.loc[tide["Time"].isin(data_frame["Time"]), "tide"].tolist()
    data_frame["Hour"] = [t.hour if t.minute == 0 else t.hour + 0.5 for t in data_frame["Time"]]

# interpolate / extrapolate

for data_frame in data_list:
    data_frame[data_frame.columns[4:]] = data_frame[data_frame.columns[4:]].interpolate("linear")
    data_frame[data_frame.columns[4:]] = data_frame[data_frame.columns[4:]].interpolate("ffill")
    data_frame[data_frame.columns[4:]] = data_frame[data_frame.columns[4:]].interpolate("bfill")

# extract daytime data

trim_list = []
for data_frame in data_list:
    date_unique = list(set(data_frame["Time"].dt.date))
    date_unique.sort()
    for d in date_unique:
        trim_list.append(data_frame.loc[data_frame["Time"].dt.date == d].
                         loc[data_frame["DayNight"]].reset_index(drop=True).copy())

########################
##### simplest ANN #####
########################

# preprocessing

X, y = [], []

for data_trimmed in trim_list:
    for i in range(data_trimmed.shape[0]):
        X.append(list(data_trimmed.iloc[i][["Temp", "dswsfc", "dlwsfc", "tcdcclm", "tmp2m", "wind", "wdir",
                                            "sst", "tide", "Hour", "Temp_diff"]]))
        if data_trimmed.iloc[i]["EAS"] == "E":
            y.append(0)
        elif data_trimmed.iloc[i]["EAS"] == "A":
            y.append(1)
        elif data_trimmed.iloc[i]["EAS"] == "S":
            y.append(2)

X = np.array(X)
y = np.array(to_categorical(y))

# plot daytime emersed temperature distribution

plot_df = pd.DataFrame(X, columns=["Temp", "dswsfc", "dlwsfc", "tcdcclm", "tmp2m", "wind", "wdir",
                                            "sst", "tide", "Hour", "Temp_diff"])
plot_df["EAS"] = [tuple(i) for i in y]
plot_df = plot_df.loc[plot_df["EAS"] == (1,0,0)]

fig, ax = plt.subplots()
ax.hist(plot_df["Temp"])

# check correlation between features

sns.pairplot(pd.DataFrame(X, columns=["Temp", "dswsfc", "dlwsfc", "tcdcclm", "tmp2m", "wind", "wdir",
                                            "sst", "tide", "Hour", "Temp_diff"]))
plt.tight_layout()

# balance training data

E_index = np.where([tuple(i) == (1, 0, 0) for i in y])[0]
A_index = np.where([tuple(i) == (0, 1, 0) for i in y])[0]
S_index = np.where([tuple(i) == (0, 0, 1) for i in y])[0]
class_weight = {0: len(y)/2/len(E_index),
                1: len(y)/2/len(A_index),
                2: len(y)/2/len(S_index)}
train_ratio = 0.8
validate_ratio = 0.2 / train_ratio

E_index_train = np.random.choice(E_index, int(len(E_index) * train_ratio), replace=False)
E_index_test = np.array([i for i in E_index if i not in E_index_train])
A_index_train = np.random.choice(A_index, int(len(A_index) * train_ratio), replace=False)
A_index_test = np.array([i for i in A_index if i not in A_index_train])
S_index_train = np.random.choice(S_index, int(len(S_index) * train_ratio), replace=False)
S_index_test = np.array([i for i in S_index if i not in S_index_train])

index_train = [E_index_train, A_index_train, S_index_train]
index_test = [E_index_test, A_index_test, S_index_test]

train_X = np.array([X[i] for list in index_train for i in list])
train_y = np.array([y[i] for list in index_train for i in list])
test_X = np.array([X[i] for list in index_test for i in list])
test_y = np.array([y[i] for list in index_test for i in list])

train_X, train_y = shuffle(train_X, train_y)

scaler = StandardScaler()
scaled_train_X = scaler.fit_transform(train_X)

# create and fit model

model = Sequential([
    Dense(units=16, activation="relu", input_shape=(11, )),
    Dense(units=16, activation="relu"),
    Dense(units=3, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

hist = History()
es = EarlyStopping(monitor="val_loss", mode="min", patience=100)
cb_list = [hist, es]

model.fit(scaled_train_X, train_y, batch_size=100, epochs=10000, verbose=2, validation_split=validate_ratio, shuffle=True,
          callbacks=cb_list, class_weight=class_weight)

# check model progress

fig, ax = plt.subplots(ncols=2)
ax[0].plot(hist.history["loss"], label="train")
ax[0].plot(hist.history["val_loss"], label="validate")
ax[0].legend()
ax[0].set_title("Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Categroical crossentropy")
ax[1].plot(np.array(hist.history["accuracy"])*100, label="train")
ax[1].plot(np.array(hist.history["val_accuracy"])*100, label="validate")
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy (%)")
ax[1].set_ylim(0, 100)
ax[1].legend()
plt.tight_layout()

# model evaluation

model.evaluate(scaler.transform(test_X), test_y)

prediction = model.predict(scaler.transform(test_X))

con_matrix = confusion_matrix(test_y.argmax(axis=1), prediction.argmax(axis=1))
ConfusionMatrixDisplay(con_matrix, display_labels=["Emersion", "Splashed", "Submersion"]).plot()

print(classification_report(test_y.argmax(axis=1), prediction.argmax(axis=1)))

# save model

os.chdir("D:\\Mirror\\mphil\\Github\\Curve-Fitting\\ANN log")

with open("simplest ANN scaler.pkl", "wb") as pickle_file:
    pickle.dump(scaler, pickle_file)

con_matrix_pd = pd.DataFrame(con_matrix)
con_matrix_pd.columns = [f"predicted {col}" for col in con_matrix_pd.columns]
con_matrix_pd.to_csv("simplest ANN con matrix.csv")

with open("simplest ANN report.txt", "a") as txt:
    txt.write(classification_report(test_y.argmax(axis=1), prediction.argmax(axis=1)))

os.chdir("D:\\Mirror\\mphil\\Github\\Curve-Fitting")

model.save("simplest ANN")

####################
##### LSTM ANN #####
####################

# preprocessing

X, y = create_LSTM_dataset(trim_list,
                      ["Temp", "dswsfc", "dlwsfc", "tcdcclm", "tmp2m", "wind", "wdir", "sst",
                       "tide", "Hour", "Temp_diff"], ["EAS"], 4, include_current_data=True)

# balance training data

E_index = np.where([tuple(i) == (1, 0, 0) for i in y])[0]
A_index = np.where([tuple(i) == (0, 1, 0) for i in y])[0]
S_index = np.where([tuple(i) == (0, 0, 1) for i in y])[0]
class_weight = {0: len(y)/2/len(E_index),
                1: len(y)/2/len(A_index),
                2: len(y)/2/len(S_index)}
train_ratio = 0.8
validate_ratio = 0.2 / train_ratio

E_index_train = np.random.choice(E_index, int(len(E_index) * train_ratio), replace=False)
E_index_test = np.array([i for i in E_index if i not in E_index_train])
A_index_train = np.random.choice(A_index, int(len(A_index) * train_ratio), replace=False)
A_index_test = np.array([i for i in A_index if i not in A_index_train])
S_index_train = np.random.choice(S_index, int(len(S_index) * train_ratio), replace=False)
S_index_test = np.array([i for i in S_index if i not in S_index_train])

index_train = [E_index_train, A_index_train, S_index_train]
index_test = [E_index_test, A_index_test, S_index_test]

train_X = np.array([X[i] for list in index_train for i in list])
train_y = np.array([y[i] for list in index_train for i in list])
test_X = np.array([X[i] for list in index_test for i in list])
test_y = np.array([y[i] for list in index_test for i in list])

train_X, train_y = shuffle(train_X, train_y)

scaler = StandardScaler()
scaled_train_X = scaler.fit_transform(train_X.reshape(train_X.shape[0] * train_X.shape[1], train_X.shape[2])).\
    reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2])

# create and fit model

model = Sequential([
    LSTM(units=16, activation="tanh", return_sequences=True, input_shape=(scaled_train_X.shape[1], scaled_train_X.shape[2])),
    LSTM(units=16, activation="tanh"),
    # LSTM(units=32, activation="tanh", return_sequences=True),
    # LSTM(units=32, activation="tanh"),
    Dense(units=3, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

hist = History()
es = EarlyStopping(monitor="val_loss", mode="min", patience=100)
cb_list = [hist, es]

model.fit(scaled_train_X, train_y, batch_size=50, epochs=5000, verbose=2, validation_split=validate_ratio, shuffle=True,
          callbacks=cb_list, class_weight=class_weight)

# check model progress

fig, ax = plt.subplots(ncols=2)
ax[0].plot(hist.history["loss"], label="train")
ax[0].plot(hist.history["val_loss"], label="validate")
ax[0].legend()
ax[0].set_title("Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Categroical crossentropy")
ax[1].plot(np.array(hist.history["accuracy"])*100, label="train")
ax[1].plot(np.array(hist.history["val_accuracy"])*100, label="validate")
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy (%)")
ax[1].set_ylim(0, 100)
ax[1].legend()
plt.tight_layout()

# model evaluation

model.evaluate(scaler.transform(test_X.reshape(test_X.shape[0] * test_X.shape[1], test_X.shape[2])).
               reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2]), test_y)

prediction = model.predict(scaler.transform(test_X.reshape(test_X.shape[0] * test_X.shape[1], test_X.shape[2])).
               reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2]))

con_matrix = confusion_matrix(test_y.argmax(axis=1), prediction.argmax(axis=1))
ConfusionMatrixDisplay(con_matrix, display_labels=["Emersion", "Splashed", "Submersion"]).plot()

print(classification_report(test_y.argmax(axis=1), prediction.argmax(axis=1)))

# save model

os.chdir("D:\\Mirror\\mphil\\Github\\Curve-Fitting\\ANN log")

with open("LSTM ANN scaler.pkl", "wb") as pickle_file:
    pickle.dump(scaler, pickle_file)

con_matrix_pd = pd.DataFrame(con_matrix)
con_matrix_pd.columns = [f"predicted {col}" for col in con_matrix_pd.columns]
con_matrix_pd.to_csv("LSTM con matrix.csv")

with open("LSTM report.txt", "a") as txt:
    txt.write(classification_report(test_y.argmax(axis=1), prediction.argmax(axis=1)))

os.chdir("D:\\Mirror\\mphil\\Github\\Curve-Fitting")

model.save("LSTM ANN")

#####################################
##### get daytime emersion time #####
#####################################

# import temperature data

temp_all = pd.read_csv("F:\\mphil\\Data\\Temperature\\combined_new.csv")
temp_all = temp_all[[col for col in temp_all.columns if re.match(r"n\ws\d", col) or col == "Time"]]
temp_all["Time"] = [datetime.strptime(t, "%m/%d/%Y %H:%M") for t in temp_all["Time"]]
temp_all["Time"] = temp_all["Time"].dt.tz_localize("Hongkong")
temp_all = temp_all.loc[(temp_all["Time"].dt.month.isin([6,7,8,9])) & (temp_all["Time"].dt.year == 2020)]

# insert DayNight column

temp_all.insert(1, "DayNight", np.full(temp_all.shape[0], np.nan))
sun = Sun(22.21, 114.26)
date_list = list(set(temp_all["Time"].dt.date))
date_list.sort()
for d in date_list:
    sunrise = sun.get_local_sunrise_time(d - timedelta(days=1))
    sunset = sun.get_local_sunset_time(d)
    temp_all.loc[temp_all["Time"].dt.date == d, "DayNight"] = \
        [sunrise < t < sunset for t in temp_all.loc[temp_all["Time"].dt.date == d, "Time"]]

# select useful EnvLoggers

nrow = 5546

data_list = []
for col in [col for col in temp_all.columns if re.match(r"n\ws\d", col)]:
    globals()[f"df_{col}"] = temp_all[["Time", "DayNight", col]]
    globals()[f"df_{col}"].columns = ["Time", "DayNight", "Temp"]
    globals()[f"df_{col}"].dropna(axis=0, inplace=True)
    if globals()[f"df_{col}"].shape[0] == nrow:
        data_list.append(globals()[f"df_{col}"])
    elif globals()[f"df_{col}"].shape[0] > nrow:
        data_list.append(globals()[f"df_{col}"].iloc[:nrow - globals()[f"df_{col}"].shape[0]])

for data_frame in data_list:
    data_frame.reset_index(drop=True, inplace=True)

# combine data

for data_frame in data_list:
    data_frame["Temp_diff"] = data_frame["Temp"].diff()
    data_frame.drop(0, inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    data_frame.loc[data_frame["Time"].isin(dswsfc["Time"]), "dswsfc"] = \
        dswsfc.loc[dswsfc["Time"].isin(data_frame["Time"]), "dswsfc"].tolist()
    data_frame.loc[data_frame["Time"].isin(dlwsfc["Time"]), "dlwsfc"] = \
        dlwsfc.loc[dlwsfc["Time"].isin(data_frame["Time"]), "dlwsfc"].tolist()
    data_frame.loc[data_frame["Time"].isin(tcdcclm["Time"]), "tcdcclm"] = \
        tcdcclm.loc[tcdcclm["Time"].isin(data_frame["Time"]), "tcdcclm"].tolist()
    data_frame.loc[data_frame["Time"].isin(tmp2m["Time"]), "tmp2m"] = \
        tmp2m.loc[tmp2m["Time"].isin(data_frame["Time"]), "tmp2m"].tolist()
    data_frame.loc[data_frame["Time"].isin(wind["Time"]), "wind"] = \
        wind.loc[wind["Time"].isin(data_frame["Time"]), "wind"].tolist()
    data_frame.loc[data_frame["Time"].isin(wdir["Time"]), "wdir"] = \
        wdir.loc[wdir["Time"].isin(data_frame["Time"]), "wdir"].tolist()
    data_frame.loc[data_frame["Time"].isin(sst["Time"]), "sst"] = \
        sst.loc[sst["Time"].isin(data_frame["Time"]), "sst"].tolist()
    data_frame.loc[data_frame["Time"].isin(tide["Time"]), "tide"] = \
        tide.loc[tide["Time"].isin(data_frame["Time"]), "tide"].tolist()
    data_frame["Hour"] = [t.hour if t.minute == 0 else t.hour + 0.5 for t in data_frame["Time"]]

# interpolate / extrapolate

for data_frame in data_list:
    data_frame[data_frame.columns[4:]] = data_frame[data_frame.columns[4:]].interpolate("linear")
    data_frame[data_frame.columns[4:]] = data_frame[data_frame.columns[4:]].interpolate("ffill")
    data_frame[data_frame.columns[4:]] = data_frame[data_frame.columns[4:]].interpolate("bfill")

# extract daytime data

trim_list = []
for data_frame in data_list:
    date_unique = list(set(data_frame["Time"].dt.date))
    date_unique.sort()
    for d in date_unique:
        trim_list.append(data_frame.loc[data_frame["Time"].dt.date == d].
                         loc[data_frame["DayNight"]].reset_index(drop=True).copy())

# load model and scaler

os.chdir("F:\\mphil\\Github\\Curve-Fitting\\ANN log")
with open("LSTM ANN scaler.pkl", "rb") as pickle_file:
    scaler = pickle.load(pickle_file)

os.chdir("F:\\mphil\\Github\\Curve-Fitting")
model = keras.models.load_model("LSTM ANN")

# preprocess daytime data

X, y = create_LSTM_dataset(trim_list,
                      ["Temp", "dswsfc", "dlwsfc", "tcdcclm", "tmp2m", "wind", "wdir", "sst",
                       "tide", "Hour", "Temp_diff"], [], 4, include_current_data=True)

X = np.array(X)

X_scaled = scaler.transform(X.reshape(X.shape[0] * X.shape[1], X.shape[2])).\
    reshape(X.shape[0], X.shape[1], X.shape[2])

# predict state with model

prediction = model.predict(X_scaled)

result = pd.DataFrame([input[-1] for input in X], columns=["Temp", "dswsfc", "dlwsfc", "tcdcclm", "tmp2m", "wind", "wdir",
                                            "sst", "tide", "Hour", "Temp_diff"])
result["State"] = [state.argmax() for state in prediction]

# get daytime emersion temperature and mode

emersed = result.loc[result["State"] == 0]
emersed_temp = emersed["Temp"].to_numpy()
pd.Series(emersed_temp).value_counts()

# plot histogram

os.chdir("F:\\mphil\\Manuscripts\\Lab heart rate\\Figures\\ai")
fig, ax = plt.subplots(figsize=(6.7, 4.8), dpi=200)
ax.hist(emersed_temp, color="grey", weights=np.ones(len(emersed_temp)) / len(emersed_temp),
        bins=np.arange(emersed_temp.min() - 0.25, emersed_temp.max() + 0.25, 0.5))
ax.set_xlabel("Rock surface temperature (°C)")
ax.set_ylabel("Proportion of time (%)")
ax.set_xlim(22.5, 47.5)
ax.set_ylim(0, 0.08)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.tight_layout()
plt.savefig("daytime_emersed_temp.pdf", transparent=True)

# how representative

half_hour = len(emersed_temp) / len(data_list)
summer_time = [datetime(2020,6,1,0,0)]
while True:
    t = summer_time[-1] + timedelta(minutes=30)
    summer_time.append(t)
    if t == datetime(2020,9,24,12,30):
        break
summer_time = pd.Series(summer_time)
summer_time = summer_time.dt.tz_localize("Hongkong")

sun = Sun(22.21, 114.26)
date_list = list(set(summer_time.dt.date))
date_list.sort()
summer_daytime = []
for d in date_list:
    sunrise = sun.get_local_sunrise_time(d - timedelta(days=1))
    sunset = sun.get_local_sunset_time(d)
    summer_daytime.extend([t for t in summer_time.loc[summer_time.dt.date == d] if sunrise < t < sunset])

rep_ratio = half_hour / len(summer_daytime)

lowest_ratio = rep_ratio * 0.98 / 1  # 98% precision, 100% recall
highest_ratio = rep_ratio * 1 / 0.93  # 100% precision, 93% recall

print(f"Calculated ratio: {rep_ratio*100:.1f}%\n"
      f"Range: {lowest_ratio*100:.1f} - {highest_ratio*100:.1f}%")

# save daytime emersion temperature

os.chdir("D:\\Mirror\\mphil\\Manuscripts\\Lab heart rate\\Figures")
pd.Series(emersed_temp).to_csv("predicted_emersed_temp.csv")