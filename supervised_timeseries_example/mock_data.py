import numpy as np
import scipy as sp
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve, KFold
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def generate_poisson_point_process(_lambda, t0, t1):

    scale = 1/_lambda

    def gen_exp():
        return np.random.exponential(scale)

    data = [t0 + gen_exp()]

    while data[-1] < t1:
        data.append(data[-1] + gen_exp())

    return data[:-1]


def generate_time_series(t0, t1, nt, event_types):

    event_type_lambdas, event_type_generator_functions = zip(*event_types)

    event_times = generate_poisson_point_process(sum(event_type_lambdas), t0, t1)
    n_events = len(event_times)
    n_event_types = len(event_types)
    likelihoods = event_type_lambdas
    sum_likelihood = sum(likelihoods)
    event_probabilities = np.array(likelihoods) / sum_likelihood
    event_type_indices = sp.stats.rv_discrete(values=(range(n_event_types), event_probabilities)).rvs(size=n_events)
    t = np.linspace(t0, t1, nt)
    events = np.array([
        event_type_generator_functions[event_type_index](t - event_time)
        for event_type_index, event_time in zip(event_type_indices, event_times)
    ]).reshape(n_events, nt)
    sum_events = np.sum(events, axis=0)
    return sum_events

def generate_time_series_collection(t0, t1, nt, n_series, event_types):
    return np.array([generate_time_series(t0, t1, nt, event_types) for _ in range(n_series)])


def psd(data):
    F = np.fft.fft(data, axis=1)
    return F.real ** 2 + F.imag ** 2


def gen_supervised_timeseries_data(series_types, training_data_size, validation_data_size, time_interval, nt, psd_flag):
    t0, t1 = time_interval
    n_series_types = len(series_types)

    training_data = np.vstack([
        generate_time_series_collection(t0, t1, nt, int(training_data_size*p_event), event_types)
        for event_types, p_event in series_types
    ])
    training_data_labels = np.array([int(training_data_size*p_event)*(i,)  for i,[event_type, p_event] in enumerate(series_types)]).reshape(-1)

    validation_data = np.vstack([
        generate_time_series_collection(t0, t1, nt, int(validation_data_size*p_event), event_types)
        for event_types, p_event in series_types
    ])

    validation_data_labels = np.array([int(validation_data_size*p_event) * (i,) for i,[event_type, p_event] in enumerate(series_types)]).reshape(-1)

    if psd_flag:
        training_data = psd(training_data)
        validation_data = psd(validation_data)

    return training_data, training_data_labels, validation_data , validation_data_labels
