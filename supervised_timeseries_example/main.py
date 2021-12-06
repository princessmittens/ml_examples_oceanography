import numpy as np

import mock_data
import models


time_interval = 0, 150 # Make a record that starts at 0s and ends at 150s
nt = 1000 # Provide number of datapoints in that time interval (sample rate is nt/time_interval)
val_size = 100 # Set the validation size of the dataset here
train_size = 10000 # Set the training size of the dataset here
p_error = 0.5 # Of the val/train sizes, set the ratio of error to no_error samples
psd_flag = True # Take the power spectral density of the timeseries data

## Play around with the train/val sample sizes, note that small samples generally yield

# Functions that generate the timeseries, and the density of the events where 0.0 is where no events take place.
series_types = [
    [
        # error
        [(0.02, lambda t: np.exp(-(t**2))),
        (0.02, lambda t: 1.0 * (np.abs(t) < 1))], p_error # probability of error
    ],
    [
        # no error
        [(0.04, lambda t: np.exp(-(t**2))),
        (0.00, lambda t: 1.0 * (np.abs(t) < 1))], 1-p_error # probability of no_error
    ]
]


def main(series_types, train_size, val_size, time_interval, nt, psd_flag):
    x_train, y_train, x_test, y_test = mock_data.gen_supervised_timeseries_data(series_types, train_size, val_size, time_interval, nt, psd_flag)
    models.run_models(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main(series_types, train_size, val_size, time_interval, nt, psd_flag)
