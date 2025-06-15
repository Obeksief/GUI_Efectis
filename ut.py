import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import base64
import time
import matplotlib.pyplot as plt
import plotly.express as px
import base64
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import tensorflow as tf
from io import BytesIO
import base64
from sklearn.multioutput import MultiOutputRegressor
import keras
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from IPython.display import clear_output, display
import optuna
import math
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score

 

TF_ENABLE_ONEDNN_OPTS=0

############################################
##       Displaying functions             ##
############################################



############################################
##         Data Cleaning Functions        ##
############################################

def split_input_output(input_data, inputs, outputs):
    X = input_data[inputs]
    y = input_data[outputs]
    return X, y

def scale_data(X, y):

    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()

    X_scaled = scaler_1.fit_transform(X)
    y_scaled = scaler_2.fit_transform(y)

    st.session_state['scaler_X'] = scaler_1
    st.session_state['scaler_y'] = scaler_2
    
    return X_scaled, y_scaled

############################################
##         Model Creating Functions       ##
############################################

def create_random_forest():
    random_forest = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                           max_depth=3,
                                           random_state=123))
    return random_forest

def create_neural_network_32_16(): # (32, 16)
    neural_network = MLPRegressor(hidden_layer_sizes=(32, 16),
                                      activation="relu",
                                      solver="adam",
                                      learning_rate="adaptive",
                                      learning_rate_init=0.1,
                                      alpha=0.0001,
                                      early_stopping=True,
                                      n_iter_no_change=50,
                                      max_iter=1000,
                                      random_state=123)
    return neural_network

def create_xgboost():
    xgboost = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100,
                                  max_depth=3,
                                  learning_rate=0.1,
                                  random_state=123))
    return xgboost
    
def create_catboost():
    catboost = MultiOutputRegressor(cb.CatBoostRegressor(iterations=100,
                                      depth=3,
                                      learning_rate=0.1,
                                      loss_function='RMSE',
                                      random_state=123))
    return catboost

### MAJ : Nouvelles fonctions de créations de modèles
###   - réseaux de neurones avec diverses architectures -
###       architecture constante : relu - adam - adaptative - lr_init 0.1
###       architecture variable : (64, 64), (128, 64), (128, 128), (128, 64, 32)

def create_neural_network_64_64(): # (64, 64)
    neural_network = MLPRegressor(hidden_layer_sizes= (64, 64),
                                      activation="relu",
                                      solver="adam",
                                      learning_rate="adaptive",
                                      learning_rate_init=0.1,
                                      alpha=0.0001,
                                      early_stopping=True,
                                      n_iter_no_change=50,
                                      max_iter=1000,
                                      random_state=123)
    return neural_network

def create_neural_network_128_64(): # (128, 64)
    neural_network = MLPRegressor(hidden_layer_sizes= (128, 64),
                                      activation="relu",
                                      solver="adam",
                                      learning_rate="adaptive",
                                      learning_rate_init=0.1,
                                      alpha=0.0001,
                                      early_stopping=True,
                                      n_iter_no_change=50,
                                      max_iter=1000,
                                      random_state=123)
    return neural_network

def create_neural_network_128_128(): # (128, 128)
    neural_network = MLPRegressor(hidden_layer_sizes= (128, 128),
                                      activation="relu",
                                      solver="adam",
                                      learning_rate="adaptive",
                                      learning_rate_init=0.1,
                                      alpha=0.0001,
                                      early_stopping=True,
                                      n_iter_no_change=50,
                                      max_iter=1000,
                                      random_state=123)
    return neural_network

def create_neural_network_128_64_32(): # (128, 64, 32)
    neural_network = MLPRegressor(hidden_layer_sizes= (128, 64, 32),
                                      activation="relu",
                                      solver="adam",
                                      learning_rate="adaptive",
                                      learning_rate_init=0.1,
                                      alpha=0.0001,
                                      early_stopping=True,
                                      n_iter_no_change=50,
                                      max_iter=1000,
                                      random_state=123)
    return neural_network

############################################
##         Model Training Functions       ##
############################################
    
def train_neural_network(neural_network, X_scaled, y_scaled):
    neural_network.fit(X_scaled, y_scaled)
    return neural_network
    
def train_xgboost(xgboost, X, y):
    xgboost.fit(X, y)
    return xgboost
    
def train_catboost(catboost, X, y):
    catboost.fit(X, y)
    return catboost
    
def train_random_forest(random_forest, X, y):
    random_forest.fit(X, y)
    return random_forest


########################################
##         Download Functions         ##
########################################


def download_scaler(scaler):
    output_scaler = pickle.dumps(scaler)
    b64 = base64.b64encode(output_scaler).decode()
    href = f'<a href="data:file/output_scaler;base64,{b64}" download="scaler.pkl">Download scaler .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)

def generate_excel_download_link(df, nom):
    towrite = BytesIO()
    df.to_excel(towrite, index=False, header=True)  # write to BytesIO buffer
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{nom}.xlsx">Download {nom} File</a>'
    return st.markdown(href, unsafe_allow_html=True)

def download_model(model, model_name):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="{model_name}.pkl">Download Trained {model_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_model_and_scalers(model,scaler_X, scaler_y, file_name):
    file = (model, scaler_X, scaler_y)
    output = pickle.dumps(file)
    b64 = base64.b64encode(output).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="modele_{file_name}.pkl">Download Trained {file_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_model_and_scalers_and_encoder(model,scaler_X, scaler_y, encoder, file_name):
    file = (model, scaler_X, scaler_y, encoder)
    output = pickle.dumps(file)
    b64 = base64.b64encode(output).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="modele_{file_name}.pkl">Download Trained {file_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_model_and_encoder(model, encoder, file_name):
    file = (model, encoder)
    output = pickle.dumps(file)
    b64 = base64.b64encode(output).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="modele_{file_name}.pkl">Download Trained {file_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_model_and_scalers_and_labels(model, scaler_X, scaler_Y, input_labels, output_labels, file_name):
    file = (model, scaler_X, scaler_Y, input_labels, output_labels)
    output = pickle.dumps(file)
    b64 = base64.b64encode(output).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="modele_{file_name}.pkl">Download Trained {file_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_model_and_scalers_and_encoder_and_labels(model,scaler_X, scaler_y, encoder,input_labels, output_labels, file_name):
    file = (model, scaler_X, scaler_y, encoder, input_labels, output_labels)
    output = pickle.dumps(file)
    b64 = base64.b64encode(output).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="modele_{file_name}.pkl">Download Trained {file_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_model_and_encoder_and_labels(model, encoder, input_labels, output_labels, file_name):
    file = (model, encoder, input_labels, output_labels)
    output = pickle.dumps(file)
    b64 = base64.b64encode(output).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="modele_{file_name}.pkl">Download Trained {file_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_model_and_labels(model, input_labels, output_labels, file_name):
    file = (model, input_labels, output_labels)
    output = pickle.dumps(file)
    b64 = base64.b64encode(output).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="modele_{file_name}.pkl">Download Trained {file_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_model_and_scalers_and_encoder_and_labels_and_params(model,scaler_X, scaler_y,params,encoder, input_labels, output_labels, file_name):
    file = (model, scaler_X, scaler_y,params, encoder, input_labels, output_labels)
    output = pickle.dumps(file)
    b64 = base64.b64encode(output).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="modele_{file_name}.pkl">Download Trained {file_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_model_and_scalers_and_labels_and_params(model,scaler_X, scaler_y, params, input_labels, output_labels, file_name):
    file = (model, scaler_X, scaler_y, params, input_labels, output_labels)
    output = pickle.dumps(file)
    b64 = base64.b64encode(output).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="modele_{file_name}.pkl">Download Trained {file_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)
    
######################################################################
##      SandBox - Entraînement d'un modèle personnalisé             ##
######################################################################

class EarlyStop_loss(tf.keras.callbacks.Callback):
    def __inti__(self, model, valid_loss_threshold):
        self.model = model
        self.valid_loss_threshold = valid_loss_threshold
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') <= self.valid_loss_threshold:
            print(f'\n\nSeuil d\'arrêt anticipé atteint. Arrêt de l\'entraînement...')
            self.model.stop_training = True

class BestModelSaver(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BestModelSaver, self).__init__()
        self.best_weights = None
        self.best_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

class Plot(keras.callbacks.Callback):

    '''Class de callback qui affiche à chaque epoch les pertes sur l'ensemble d'entrainement 
       et sur l'ensemble de validation, ce qui permet de suivre l'entrainement en temps réelle'''
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.index = 0
        self.fig = plt.figure()
        self.logs = []
        self.echelle = 10
    
    def on_epoch_begin(self, epoch, logs=None):
        
        # Si la perte actuelle est 10x plus petite que la première perte affichée
        # on met à jour le graphique en mettant en première perte la perte actuelle
        if epoch > 2 and self.losses[0]/self.echelle > self.losses[self.i-1] :
            self.i = 0
            self.losses = []
            self.val_losses = []
            self.x = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.index)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        self.index += 1 
        
        clear_output(wait=True)
        

        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        st.pyplot(self.fig, clear_figure=True, use_container_width=True)
        
def create_customized_model(input_shape, list_neurons, liste_activation, output_shape):
    model = Sequential()
    model.add(keras.layers.Dense(list_neurons[0], input_shape=(input_shape,), activation=liste_activation[0]))
    for i in range(1, len(list_neurons)):
        model.add(Dense(list_neurons[i], activation=liste_activation[i]))
    model.add(Dense(output_shape, activation='linear'))
    return model

def compile_customized_model(model, learning_rate_init, optimizer, loss, metrics):
    if optimizer == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate_init)
    elif optimizer == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate_init)
    elif optimizer == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate_init)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

def preprocess_data(X_num, Y, scaler):

    if scaler == 'StandardScaler':
        sc_X = StandardScaler()
        sc_Y = StandardScaler()
    elif scaler == 'MinMaxScaler':
        sc_X = MinMaxScaler()
        sc_Y = MinMaxScaler()
    elif scaler == 'RobustScaler':
        sc_X = RobustScaler()
        sc_Y = RobustScaler()
    
    X_num_scaled = sc_X.fit_transform(X_num)
    Y_scaled = sc_Y.fit_transform(Y)


    return X_num_scaled, Y_scaled, sc_X, sc_Y

def create_customized_callbacks(metrics, dic_seuil_early_stopping, cyclic, display_graph):
    callbacks = []
    return callbacks

def train_customized_model(model, X_train, Y_train, epochs, batch_size,valid_size):
    save_best = BestModelSaver()
    plot = Plot()
    history = model.fit(X_train, 
              Y_train, 
              epochs=epochs, 
              batch_size=batch_size, 
              validation_split=valid_size,
                callbacks=[save_best],
              verbose=0)
    return model, history 

def evaluate_customized_model(model, X_test, Y_test):
    # https://keras.io/api/models/model_training_apis/#evaluate-method
    err = model.evaluate(X_test, Y_test)

    return err

############################################################################
##      Sandbox - Entraînement par plage d'hyperparamètres                ##
############################################################################

### NN related functions
def get_radar_nn_optim():
    a_scaled = math.log2(st.session_state['best_params']['first_layer'])/math.log2(128)
    b_scaled = math.log2(st.session_state['best_params']['second_layer'])-1/math.log2(128)-1
    c_scaled = math.log10(st.session_state['best_params']['batch_size'])/math.log10(100)

    fig_radar_optim = go.Figure()

    fig_radar_optim.add_trace(go.Scatterpolar(
    r=[ a_scaled,
                        b_scaled,
                        c_scaled],
    theta=st.session_state['categories'],
    fill='toself',
    name='Hyperparamètres optimales'))

    fig_radar_optim.update_layout(
                        title='Hyperparamètres optimales',
                    polar=dict(
                            radialaxis=dict(
                            visible=True,
                            range=[0, 1])),
                        showlegend=False)
    
    st.plotly_chart(fig_radar_optim)

def get_radar_nn_slider():
    min_a = st.session_state['slider_range_first_layer'][0]
    max_a = st.session_state['slider_range_first_layer'][1]
    min_b = st.session_state['slider_range_second_layer'][0]
    max_b = st.session_state['slider_range_second_layer'][1]
    min_c = st.session_state['slider_range_batch_size'][0]
    max_c = st.session_state['slider_range_batch_size'][1]

    min_a_prescaled = math.log2(min_a)
    max_a_prescaled = math.log2(max_a)
    min_b_prescaled = math.log2(min_b)
    max_b_prescaled = math.log2(max_b)


    min_a_scaled = min_a_prescaled/7
    max_a_scaled = max_a_prescaled/7
    min_b_scaled = min_b_prescaled/7
    max_b_scaled = max_b_prescaled/7
    min_c_scaled = math.log10(min_c)/math.log10(100)
    max_c_scaled = math.log10(max_c)/math.log10(100)

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
                    r=[ min_a_scaled,
                        min_b_scaled,
                        min_c_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne minimum'
                ))
    
    fig_radar.add_trace(go.Scatterpolar(
                    r=[ max_a_scaled,
                        max_b_scaled,
                        max_c_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne max'
                ))
    
    fig_radar.update_layout(
                title = 'Hyperparamètres',
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    )),
                showlegend=False
                )
    
    st.plotly_chart(fig_radar)

def objective_nn(trial):  
    
    range_layer_1 = liste_puissance_de_deux(st.session_state['slider_range_first_layer'][0], st.session_state['slider_range_first_layer'][1])
    range_layer_2 = liste_puissance_de_deux(st.session_state['slider_range_second_layer'][0], st.session_state['slider_range_second_layer'][1])
    range_batch_size = dizaines(st.session_state['slider_range_batch_size'][0], st.session_state['slider_range_batch_size'][1])
    first_layer = trial.suggest_categorical('first_layer', range_layer_1)
    second_layer = trial.suggest_categorical('second_layer', range_layer_2)
    batch_size = trial.suggest_categorical('batch_size', range_batch_size)

    model = MLPRegressor(hidden_layer_sizes=[first_layer, second_layer],
                         batch_size=batch_size,
                            activation='relu',
                            solver='adam',
                            alpha=0.0001,
                            learning_rate='adaptive',
                            learning_rate_init=0.1,
                            beta_1=0.9,
                            beta_2=0.999,
                            max_iter=st.session_state['nb_epoch'],
                            validation_fraction=0.15,
                            early_stopping=True,
                            tol=0.0001,
                            n_iter_no_change=20)
    score = cross_val_score(model, st.session_state['X_scaled'], st.session_state['y'], cv=3, scoring='neg_mean_squared_error')

    ############# Progress bar test
    st.session_state['my_bar'].progress(trial.number/st.session_state['nb_trials'])
    #############
    

    return score.mean()

def launch_optim_nn():
    ############# Progress bar test
    progress_text = "Calcul en cours. Veuillez patienter."
    st.session_state['my_bar'] = st.progress(0, text=progress_text)
    #############
    

    with st.spinner('Optimisation des hyperparamètres...'):
        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
        study.optimize(objective_nn, n_trials=st.session_state['nb_trials'])
        ############# Progress bar test
        st.session_state['my_bar'].empty()
        #############
        st.session_state['best_params'] = study.best_params
        st.plotly_chart(optuna.visualization.plot_parallel_coordinate(study)) 
    

    
  
### xgboost related functions
def get_radar_xgboost_optim():
    a_scaled = (st.session_state['best_params']['n_estimators'] -20)/(500-20)
    b_scaled = (st.session_state['best_params']['max_depth'] -1)/(15-1)
    c_scaled = (st.session_state['best_params']['eta'] -0.01)/(0.5-0.01)
    d_scaled = (st.session_state['best_params']['min_child_weight'] -1)/(10-1)
    fig_radar_optim = go.Figure()

    fig_radar_optim.add_trace(go.Scatterpolar(
    r=[ a_scaled,
                        b_scaled,
                        c_scaled,
                        d_scaled],
    theta=st.session_state['categories'],
    fill='toself',
    name='Hyperparamètres optimales'))

    fig_radar_optim.update_layout(
                        title='Hyperparamètres optimales',
                    polar=dict(
                            radialaxis=dict(
                            visible=True,
                            range=[0, 1])),
                        showlegend=False)

    st.plotly_chart(fig_radar_optim)

    st.markdown('#### Les hyperparamètres optimaux sont :')
    st.write('nombre d estimateurs :', st.session_state['best_params']['n_estimators'])
    st.write('profondeur maximale :', st.session_state['best_params']['max_depth'])
    st.write('taux d apprentissage :', round(st.session_state['best_params']['eta'],2))
    st.write('poids minimal des feuilles :', st.session_state['best_params']['min_child_weight'])

def get_radar_xgboost_slider():
    min_a = st.session_state['slider_range_nbr_estimateurs'][0]
    max_a = st.session_state['slider_range_nbr_estimateurs'][1]
    min_b = st.session_state['slider_range_max_depth'][0]
    max_b = st.session_state['slider_range_max_depth'][1]
    min_c = st.session_state['slider_range_eta'][0]
    max_c = st.session_state['slider_range_eta'][1]
    min_d = st.session_state['slider_range_min_child_weight'][0]
    max_d = st.session_state['slider_range_min_child_weight'][1]

    min_a_scaled = (min_a -20)/(500-20)
    max_a_scaled = (max_a -20)/(500-20)
    min_b_scaled = (min_b -1)/(15-1)
    max_b_scaled = (max_b -1)/(15-1)
    min_c_scaled = (min_c -0.01)/(0.5-0.01)
    max_c_scaled = (max_c -0.01)/(0.5-0.01)
    min_d_scaled = (min_d -1)/(10-1)
    max_d_scaled = (max_d -1)/(10-1)

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
                    r=[ min_a_scaled,
                        min_b_scaled,
                        min_c_scaled,
                        min_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne minimum'
                ))
    fig_radar.add_trace(go.Scatterpolar(
                    r=[ max_a_scaled,
                        max_b_scaled,
                        max_c_scaled,
                        max_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne max'
                ))

    fig_radar.update_layout(
                title = 'Hyperparamètres',
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    )),
                showlegend=False
                )

    st.plotly_chart(fig_radar)

def objective_xgboost(trial):
    n_estimators = trial.suggest_int('n_estimators', st.session_state['range_nbr_estimateurs'][0], st.session_state['range_nbr_estimateurs'][1])
    max_depth = trial.suggest_int('max_depth', st.session_state['range_max_depth'][0], st.session_state['range_max_depth'][1])
    eta = trial.suggest_float('eta', st.session_state['range_eta'][0], st.session_state['range_eta'][1])
    min_child_weight = trial.suggest_int('min_child_weight', st.session_state['range_min_child_weight'][0]  , st.session_state['range_min_child_weight'][1])

    model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth,
                             eta=eta,
                             min_child_weight=min_child_weight,
                             random_state=123))

    score = cross_val_score(model, st.session_state['X_train_scaled'], st.session_state['y_train_scaled'], cv=3, scoring='neg_mean_squared_error')

    ############# Progress bar test
    st.session_state['my_bar'].progress(trial.number/st.session_state['nb_trials'])
    #############
    
    return score.mean()

def launch_optim_xgboost():
                    ############# Progress bar test
                    progress_text = "Calcul en cours. Veuillez patienter."
                    st.session_state['my_bar'] = st.progress(0, text=progress_text)
                    #############
                    st.session_state['range_nbr_estimateurs'] = st.session_state['slider_range_nbr_estimateurs']
                    st.session_state['range_max_depth'] = st.session_state['slider_range_max_depth']
                    st.session_state['range_eta'] = st.session_state['slider_range_eta']
                    st.session_state['range_min_child_weight'] = st.session_state['slider_range_min_child_weight']


                    with st.spinner('Optimisation des hyperparamètres...'):
                        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
                        study.optimize(objective_xgboost,
                                       n_trials=st.session_state['nb_trials'],
                                       timeout = st.session_state['temps_max'])
                        ############# Progress bar test
                        st.session_state['my_bar'].empty()
                        #############
                        st.session_state['best_params'] = study.best_params
                        st.write(type(study.best_params)) ### A Supprimer
                        st.plotly_chart(optuna.visualization.plot_parallel_coordinate(study)) 
                    


### Random Forest related functions
def objective_random_forest(trial):
    n_estimators = trial.suggest_int('n_estimators', st.session_state['range_n_estimators'][0], st.session_state['range_n_estimators'][1])
    max_depth = trial.suggest_int('max_depth', st.session_state['range_max_depth'][0], st.session_state['range_max_depth'][1])
    min_samples_split = trial.suggest_int('min_samples_split', st.session_state['range_min_samples_split'][0], st.session_state['range_min_samples_split'][1])
    min_samples_leaf = trial.suggest_int('min_samples_leaf', st.session_state['range_min_samples_leaf'][0], st.session_state['range_min_samples_leaf'][1])

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=123))

    score = cross_val_score(model, st.session_state['X'], st.session_state['y'], cv=3, scoring='neg_mean_squared_error')


    ############# Progress bar test
    st.session_state['my_bar'].progress(trial.number/st.session_state['nb_trials'])
    #############
    
    
    return score.mean()

def launch_optim_random_forest():
    ############# Progress bar test
    progress_text = "Calcul en cours. Veuillez patienter."
    st.session_state['my_bar'] = st.progress(0, text=progress_text)
    #############
    #st.session_state['range_n_estimators'] = st.session_state['slider_range_n_estimators']
    #st.session_state['range_max_depth'] = st.session_state['slider_range_max_depth']
    #st.session_state['range_min_samples_split'] = st.session_state['slider_range_min_samples_split']
    #st.session_state['range_min_samples_leaf'] = st.session_state['slider_range_min_samples_leaf']

    st.session_state['slider_range_n_estimators'] = st.session_state['range_n_estimators'] 
    st.session_state['slider_range_max_depth'] = st.session_state['range_max_depth']
    st.session_state['slider_range_min_samples_split'] = st.session_state['range_min_samples_split'] 
    st.session_state['slider_range_min_samples_leaf'] = st.session_state['range_min_samples_leaf'] 

    with st.spinner('Optimisation des hyperparamètres...'):
        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
        study.optimize(objective_random_forest, n_trials=st.session_state['nb_trials'])
        ############# Progress bar test
        st.session_state['my_bar'].empty()
        #############
        st.session_state['best_params'] = study.best_params
        st.plotly_chart(optuna.visualization.plot_parallel_coordinate(study)) 
        
def get_radar_random_forest_slider():
    min_a = st.session_state['slider_range_n_estimators'][0]
    max_a = st.session_state['slider_range_n_estimators'][1]
    min_b = st.session_state['slider_range_max_depth'][0]
    max_b = st.session_state['slider_range_max_depth'][1]
    min_c = st.session_state['slider_range_min_samples_split'][0]
    max_c = st.session_state['slider_range_min_samples_split'][1]
    min_d = st.session_state['slider_range_min_samples_leaf'][0]
    max_d = st.session_state['slider_range_min_samples_leaf'][1]

    min_a_scaled = (min_a -10)/(500-10)
    max_a_scaled = (max_a -10)/(500-10)
    min_b_scaled = (min_b -1)/(50-1)
    max_b_scaled = (max_b -1)/(50-1)
    min_c_scaled = (min_c -2)/(10-2)
    max_c_scaled = (max_c -2)/(10-2)
    min_d_scaled = (min_d -1)/(10-1)
    max_d_scaled = (max_d -1)/(10-1)

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
                    r=[ min_a_scaled,
                        min_b_scaled,
                        min_c_scaled,
                        min_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne minimum'
                ))
    fig_radar.add_trace(go.Scatterpolar(
                    r=[ max_a_scaled,
                        max_b_scaled,
                        max_c_scaled,
                        max_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne max'
                ))

    fig_radar.update_layout(
                title = 'Hyperparamètres',
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    )),
                showlegend=False
                )

    st.plotly_chart(fig_radar)

def get_radar_random_forest_optim():
    a_scaled = (st.session_state['best_params']['n_estimators'] -10)/(500-10)
    b_scaled = (st.session_state['best_params']['max_depth'] -1)/(50-1)
    c_scaled = (st.session_state['best_params']['min_samples_split'] -2)/(10-2)
    d_scaled = (st.session_state['best_params']['min_samples_leaf'] -1)/(10-1)
    fig_radar_optim = go.Figure()

    fig_radar_optim.add_trace(go.Scatterpolar(
    r=[ a_scaled,
                        b_scaled,
                        c_scaled,
                        d_scaled],
    theta=st.session_state['categories'],
    fill='toself',
    name='Hyperparamètres optimales'))

    fig_radar_optim.update_layout(
                        title='Hyperparamètres optimales',
                    polar=dict(
                            radialaxis=dict(
                            visible=True,
                            range=[0, 1])),
                        showlegend=False)

    st.plotly_chart(fig_radar_optim)


### CatBoost related functions
def get_radar_catboost_optim():
    a_scaled = (st.session_state['best_params']['n_iterations'] -50)/(1000-50)
    b_scaled = (st.session_state['best_params']['learning_rate'] -0.001)/(0.5-0.001)
    c_scaled = (st.session_state['best_params']['depth'] -1)/(15-1)
    d_scaled = (st.session_state['best_params']['subsample'] -0.05)/(1-0.05)
    fig_radar_optim = go.Figure()

    fig_radar_optim.add_trace(go.Scatterpolar(
    r=[ a_scaled,
                        b_scaled,
                        c_scaled,
                        d_scaled],
    theta=st.session_state['categories'],
    fill='toself',
    name='Hyperparamètres optimales'))

    fig_radar_optim.update_layout(
                        title='Hyperparamètres optimales',
                    polar=dict(
                            radialaxis=dict(
                            visible=True,
                            range=[0, 1])),
                        showlegend=False)

    st.plotly_chart(fig_radar_optim)

def get_radar_catboost_slider():
    min_a = st.session_state['slider_range_n_iterations'][0]
    max_a = st.session_state['slider_range_n_iterations'][1]
    min_b = st.session_state['slider_range_learning_rate'][0]
    max_b = st.session_state['slider_range_learning_rate'][1]
    min_c = st.session_state['slider_range_depth'][0]
    max_c = st.session_state['slider_range_depth'][1]
    min_d = st.session_state['slider_range_subsample'][0]
    max_d = st.session_state['slider_range_subsample'][1]

    min_a_scaled = (min_a -50)/(1000-50)
    max_a_scaled = (max_a -50)/(1000-50)
    min_b_scaled = (min_b -0.001)/(0.5-0.001)
    max_b_scaled = (max_b -0.001)/(0.5-0.001)
    min_c_scaled = (min_c -1)/(15-1)
    max_c_scaled = (max_c -1)/(15-1)
    min_d_scaled = (min_d -0.05)/(1-0.05)
    max_d_scaled = (max_d -0.05)/(1-0.05)

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
                    r=[ min_a_scaled,
                        min_b_scaled,
                        min_c_scaled,
                        min_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne minimum'
                ))
    fig_radar.add_trace(go.Scatterpolar(
                    r=[ max_a_scaled,
                        max_b_scaled,
                        max_c_scaled,
                        max_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne max'
                ))

    fig_radar.update_layout(
                title = 'Hyperparamètres',
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    )),
                showlegend=False
                )

    st.plotly_chart(fig_radar)

def objective_catboost(trial):
    n_iterations = trial.suggest_int('n_iterations', st.session_state['range_n_iterations'][0], st.session_state['range_n_iterations'][1])
    learning_rate = trial.suggest_float('learning_rate', st.session_state['range_learning_rate'][0], st.session_state['range_learning_rate'][1])
    depth = trial.suggest_int('depth', st.session_state['range_depth'][0], st.session_state['range_depth'][1])
    subsample = trial.suggest_float('subsample', st.session_state['range_subsample'][0], st.session_state['range_subsample'][1])

    model = MultiOutputRegressor(cb.CatBoostRegressor(iterations=n_iterations,
                              learning_rate=learning_rate,
                              depth=depth,
                              subsample=subsample,
                              random_state=123))

    score = cross_val_score(model, st.session_state['X_scaled'], st.session_state['y_scaled'], cv=3, scoring='neg_mean_squared_error', verbose=0)


    ############# Progress bar test
    st.session_state['my_bar'].progress(trial.number/st.session_state['nb_trials'])
    #############
    
    
    return score.mean()

def launch_optim_catboost():
    ## Progress bar 
    progress_text = "Calcul en cours. Veuillez patienter."
    st.session_state['my_bar'] = st.progress(0, text=progress_text)
    ##
    st.session_state['range_n_iterations'] = st.session_state['slider_range_n_iterations']
    st.session_state['range_learning_rate'] = st.session_state['slider_range_learning_rate']
    st.session_state['range_depth'] = st.session_state['slider_range_depth']
    st.session_state['range_subsample'] = st.session_state['slider_range_subsample']
    #st.session_state['nb_trials'] = nb_trial

    with st.spinner('Optimisation des hyperparamètres...'):
        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
        study.optimize(objective_catboost, n_trials=st.session_state['nb_trials'])
        ## Progress bar 
        st.session_state['my_bar'].empty()
        ##
        st.session_state['best_params'] = study.best_params
        st.plotly_chart(optuna.visualization.plot_parallel_coordinate(study)) 


### Autres fonctions

def liste_puissance_de_deux(min, max):
    """
    Génère une liste des puissances de deux dans une plage spécifiée.

    Cette fonction renvoie une liste contenant toutes les puissances de deux qui sont
    comprises entre `min` (inclus si c'est une puissance de deux) et `max` (inclus
    également si c'est une puissance de deux). Elle commence par trouver la première
    puissance de deux supérieure ou égale à `min`, puis continue jusqu'à ce qu'elle
    dépasse `max`.

    Paramètres
    ----------
    min : int
        La borne inférieure de la plage. La première puissance de deux trouvée sera
        supérieure ou égale à ce nombre.
    max : int
        La borne supérieure de la plage. Aucune puissance de deux supérieure à ce
        nombre ne sera ajoutée à la liste.

    Retourne
    -------
    list
        Une liste contenant les puissances de deux dans la plage spécifiée.

    Exemple
    -------
    >>> liste_puissance_de_deux(5, 100)
    [8, 16, 32, 64]
    """

    puissance = 1  # Initialisation de la première puissance de deux

    # Trouver la première puissance de deux >= min
    while puissance < min:
        puissance *= 2

    puissances = []  # Liste pour stocker les puissances de deux dans la plage

    # Ajouter les puissances de deux jusqu'à atteindre ou dépasser 'max'
    while puissance <= max:
        puissances.append(puissance)
        puissance *= 2

    return puissances


def dizaines(debut, fin):
    """
    Génère une liste de nombres représentant les dizaines entre deux bornes spécifiées.

    Cette fonction crée une liste de nombres qui correspondent aux dizaines entières
    comprises entre `debut` et `fin`, inclusivement. Si `debut` est inférieur à 10, il
    sera ajouté tel quel au début de la liste. Ensuite, la fonction continue à ajouter
    les multiples de 10 jusqu'à atteindre ou dépasser `fin`.

    Paramètres
    ----------
    debut : int
        La borne de départ de la plage, qui peut être inférieure à 10.
    fin : int
        La borne de fin de la plage.

    Retourne
    -------
    list
        Une liste de nombres entiers correspondant aux dizaines dans la plage spécifiée.
    
    Exemple
    -------
    >>> dizaines(5, 35)
    [5, 10, 20, 30]
    """

    dizaines_list = []  # Initialisation de la liste qui contiendra les dizaines

    # Si 'debut' est inférieur à 10, on l'ajoute directement à la liste et on commence la dizaine suivante à 10
    if debut < 10:
        dizaines_list.append(debut)
        debut = 10

    # Calcul de la première dizaine qui est supérieure ou égale à 'debut'
    dizaine = (debut // 10) * 10
    if dizaine < debut:
        dizaine += 10

    # Boucle pour ajouter chaque multiple de 10 jusqu'à la borne 'fin'
    while dizaine <= fin:
        dizaines_list.append(dizaine)
        dizaine += 10

    return dizaines_list


