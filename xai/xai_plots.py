import numpy as np
import datetime as dt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utils.utils import read_all_properties
from utils.constants import ROOT_DIRECTORY

number_to_day_string = {
    0: "Mo",
    1: "Di",
    2: "Mi",
    3: "Do",
    4: "Fr",
    5: "Sa",
    6: "So"
}


def create_plot_shap(shap_values,property_name, index):
    energy_consumption_of_household = get_original_test_data(property_name)[index]
    dates = [dt.strftime('%H:%M Z') for dt in 
       datetime_range(datetime(2016, 9, 1, 0), datetime(2016, 9, 2, 0), 
       timedelta(minutes=30))]

    print(len(dates))

    print(len(energy_consumption_of_household))
    fig = plt.figure()

    for day in range(7):
        ax1 = fig.add_subplot(711 + day)
        ax1.plot(dates[:47], energy_consumption_of_household[day*48:(day+1)*48 - 1])
        ax1.set_ylabel(number_to_day_string[day], rotation=0, labelpad=20)
        ax = plt.gca()
        plt.setp(ax1, ylim=(0,3))
        for index, label in enumerate(ax.get_xaxis().get_ticklabels()[:]):
            if index % 10 != 0:
                label.set_visible(False)
        for index,item in enumerate(shap_values):
                if item > 0.02 and index > day*48 and index < (day+1)*48 - 1:
                    ax1.axvspan((index%48) - 1,index%48, color='red')
        for index,item in enumerate(shap_values):
                if item < -0.02 and index > day*48 and index < (day+1)*48 - 1:
                    ax1.axvspan((index%48) -1,index%48, color='blue')
    plt.show()
    plt.savefig("plot_shap.png")


def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

def get_original_test_data(property_name):
    root_dir = ROOT_DIRECTORY
    datasets_dict = read_all_properties(root_dir, False)
    x_test = datasets_dict[property_name][2]

    if len(x_test.shape) == 2:  # if univariate
            #add a dimension to make it multivariate with one dimension
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_test