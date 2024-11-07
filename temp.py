def get_MAPE_and_RMSE(y_true, X_test, scaler_y, model):

    y_pred = model.predict(X_test)

    ##################
    ## Mono output  ##
    ##################
    if len(st.session_state['outputs']) == 1:
              
        # Test des modèles avec données mises à l'échelle
        if model == 'Neural Network' or model == 'XGBoost' or model == 'CatBoost':
            y_pred_inverse_scaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
            y_test_inverse_scaled = scaler_y.inverse_transform(y_true.reshape(-1, 1))

                       
            mape_err = round(mean_absolute_percentage_error(y_test_inverse_scaled, y_pred_inverse_scaled) * 100, 2)
            rmse_err = round(np.sqrt(mean_squared_error(y_test_inverse_scaled, y_pred_inverse_scaled)), 4)

        # Test des autres modèles sans mise à l'échelle
        else:

            mape_err = round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2)
            rmse_err = round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)

        return mape_err, rmse_err

    ####################
    ##  Multi output  ##
    ####################
    elif len(st.session_state['outputs']) > 1:
                
        # Test des modèles avec données mises à l'échelle
        if model == 'Neural Network' or model == 'XGBoost' or model == 'CatBoost':
            y_pred_inverse_scaled = scaler_y.inverse_transform(y_pred)
            y_test_inverse_scaled = scaler_y.inverse_transform(y_true)

            mape_err = round(mean_absolute_percentage_error(y_test_inverse_scaled, y_pred_inverse_scaled) * 100, 2)
            rmse_err = round(np.sqrt(mean_squared_error(y_test_inverse_scaled, y_pred_inverse_scaled)), 4)

            mape_err_rawvalues  = mean_absolute_percentage_error(y_test_inverse_scaled, y_pred_inverse_scaled, multioutput='raw_values') * 100
            rmse_err_rawvalues  = np.sqrt(mean_squared_error(y_test_inverse_scaled, y_pred_inverse_scaled,multioutput='raw_values'))


        # Test des autres modèles sans mise à l'échelle
        else:
            
            mape_err = round(mean_absolute_percentage_error(st.session_state['y_test'], y_pred) * 100, 2)
            rmse_err = round(np.sqrt(mean_squared_error(st.session_state['y_test'], y_pred)), 4)

            mape_err_rawvalues  = mean_absolute_percentage_error(st.session_state['y_test'], y_pred, multioutput='raw_values') * 100
            rmse_err_rawvalues  = np.sqrt(mean_squared_error(st.session_state['y_test'], y_pred, multioutput='raw_values'))

        return mape_err, rmse_err, mape_err_rawvalues, rmse_err_rawvalues

     
    ##################
    ##  No output   ##
    ##################
    else:
        st.error('Pas de données de sortie')
        return None, None
            
    
##################################
### CatBoost related functions ###
##################################
def objective_catboost(trial):

    ### Recehrche des hyperparamètres
    n_iterations = trial.suggest_int('n_iterations', st.session_state['range_n_iterations'][0], st.session_state['range_n_iterations'][1])
    learning_rate = trial.suggest_float('learning_rate', st.session_state['range_learning_rate'][0], st.session_state['range_learning_rate'][1])
    depth = trial.suggest_int('depth', st.session_state['range_depth'][0], st.session_state['range_depth'][1])
    subsample = trial.suggest_float('subsample', st.session_state['range_subsample'][0], st.session_state['range_subsample'][1])

    model = MultiOutputRegressor(cb.CatBoostRegressor(iterations=n_iterations,
                                    learning_rate=learning_rate,
                                    depth=depth,
                                    subsample=subsample,
                                    random_state=123,
                                    logging_level='Silent'))
    
    score = cross_val_score(model, st.session_state['X_train_scaled'], st.session_state['y_train_scaled'], cv=3, scoring='neg_mean_squared_error')

    ### Score MAPE pour arrêt anticipé
    # Entraînement
    model = MultiOutputRegressor(cb.CatBoostRegressor(iterations=n_iterations,
                                    learning_rate=learning_rate,
                                    depth=depth,
                                    subsample=subsample,
                                    random_state=123,
                                    logging_level='Silent'))
    model.fit(st.session_state['X_train_scaled'], st.session_state['y_train_scaled'], verbose=False)

    # Erreur MAPE et RMSE
    if len(st.session_state['outputs']) == 1:
        mape_error, rmse_error = get_MAPE_and_RMSE(st.session_state['y_test_scaled'], st.session_state['X_test_scaled'], st.session_state['scaler_y'], model)
    else:
        mape_error, rmse_error, mape_error_rawvalues, rmse_error_rawvalues = get_MAPE_and_RMSE(st.session_state['y_test_scaled'], st.session_state['X_test_scaled'], st.session_state['scaler_y'], model)


    ### Sauvegarde du meilleur modèle et les scores RMSE et MAPE
    if trial.number == 0:
        st.session_state['model'] = model
        st.session_state['model_MAPE'] = mape_error
        st.session_state['model_RMSE'] = rmse_error
        if len(st.session_state['outputs']) > 1:
            st.session_state['model_MAPE_rawvalues'] = mape_error_rawvalues
            st.session_state['model_RMSE_rawvalues'] = rmse_error_rawvalues
    if rmse_error < st.session_state['model_RMSE']:
        st.session_state['model'] = model
        st.session_state['model_MAPE'] = mape_error
        st.session_state['model_RMSE'] = rmse_error
        if len(st.session_state['outputs']) > 1:
            st.session_state['model_MAPE_rawvalues'] = mape_error_rawvalues
            st.session_state['model_RMSE_rawvalues'] = rmse_error_rawvalues

    ### Arrêt anticipé
    if mape_error < st.session_state['mape_tolerance']:
        trial.study.stop()
        st.success('Arrêt anticipé')

    ### Progress bar 
    temps_chargement = (time.time()-st.session_state['start_time'])/st.session_state['temps_max']
    #st.info(temps_chargement)
    if temps_chargement > 1:
        temps_chargement = 1
    st.session_state['my_bar'].progress(temps_chargement, text=f'Calcul en cours. Veuillez patienter. Essai n°{trial.number+1}')
    

            
            
    return score.mean()

def launch_optim_catboost():
    ### Initialisation Progress bar 
    progress_text = "Calcul en cours. Veuillez patienter."
    st.session_state['my_bar'] = st.progress(0, text=progress_text)
    ###
    ### Intialisation du temps
    st.session_state['start_time'] = time.time()
    ###

    with st.spinner(st.session_state['spinner_text_modele_optimise']):
        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
        st.info(st.session_state['temps_max'])
        study.optimize(objective_catboost, n_trials=st.session_state['nb_trials'], timeout=st.session_state['temps_max'])
        ### Progress# bar
        st.session_state['my_bar'].empty()
        ###
        st.session_state['best_params'] = study.best_params
        st.session_state['afficher_radar_param_optim'] = True

############################
### NN related functions ###
############################

def objective_nn(trial):  
    
    ### Recehrche des hyperparamètres
    first_layer = trial.suggest_categorical('first_layer', st.session_state['range_first_layer'])
    second_layer = trial.suggest_categorical('second_layer', st.session_state['range_second_layer'])
    batch_size = trial.suggest_categorical('batch_size', st.session_state['range_batch_size'])
    learning_rate_initial = trial.suggest_float('learning_rate_initial', st.session_state['learning_rate_initial'][0],
                                                st.session_state['learning_rate_initial'][1])

    model = MLPRegressor(hidden_layer_sizes=[first_layer, second_layer],
                         batch_size=batch_size,
                            activation='relu',
                            solver='adam',
                            alpha=0.0001,
                            learning_rate='adaptive',
                            learning_rate_init = learning_rate_initial,
                            beta_1=0.9,
                            beta_2=0.999,
                            max_iter=1000,
                            validation_fraction=0.15,
                            early_stopping=True,
                            tol=0.0001,
                            n_iter_no_change=20)
    
    score = cross_val_score(model, st.session_state['X_train_scaled'], st.session_state['y_train_scaled'], cv=3, scoring='neg_mean_squared_error')

    #### Score MAPE pour arrêt anticipé
    # Entraînement
    model = MLPRegressor(hidden_layer_sizes=[first_layer, second_layer],
                         batch_size=batch_size,
                            activation='relu',
                            solver='adam',
                            alpha=0.0001,
                            learning_rate='adaptive',
                            learning_rate_init = learning_rate_initial,
                            beta_1=0.9,
                            beta_2=0.999,
                            max_iter=1000,
                            validation_fraction=0.15,
                            early_stopping=True,
                            tol=0.0001,
                            n_iter_no_change=20)
    
    model.fit(st.session_state['X_train_scaled'], st.session_state['y_train_scaled'])


    # Erreur MAPE et RMSE
    # Erreur MAPE et RMSE
    if len(st.session_state['outputs']) == 1:
        mape_error, rmse_error = get_MAPE_and_RMSE(st.session_state['y_test_scaled'], st.session_state['X_test_scaled'], st.session_state['scaler_y'], model)
    else:
        mape_error, rmse_error, mape_error_rawvalues, rmse_error_rawvalues = get_MAPE_and_RMSE(st.session_state['y_test_scaled'], st.session_state['X_test_scaled'], st.session_state['scaler_y'], model)

    ### Sauvegarde du meilleur modèle et les scores RMSE et MAPE
    if trial.number == 0:
        st.session_state['model'] = model
        st.session_state['model_MAPE'] = mape_error
        st.session_state['model_RMSE'] = rmse_error
        if len(st.session_state['outputs']) > 1:
            st.session_state['model_MAPE_rawvalues'] = mape_error_rawvalues
            st.session_state['model_RMSE_rawvalues'] = rmse_error_rawvalues
    if rmse_error < st.session_state['model_RMSE']:
        st.session_state['model'] = model
        st.session_state['model_MAPE'] = mape_error
        st.session_state['model_RMSE'] = rmse_error
        if len(st.session_state['outputs']) > 1:
            st.session_state['model_MAPE_rawvalues'] = mape_error_rawvalues
            st.session_state['model_RMSE_rawvalues'] = rmse_error_rawvalues

    ### Arrêt anticipé
    if mape_error < st.session_state['mape_tolerance']:
        trial.study.stop()
        st.success('Arrêt anticipé')

    ### Progress bar 
    temps_chargement = (time.time()-st.session_state['start_time'])/st.session_state['temps_max']
    if temps_chargement > 1:
        temps_chargement = 1
    st.session_state['my_bar'].progress(temps_chargement, text=f'Calcul en cours. Veuillez patienter. Essai n°{trial.number+1}')
    #
    return score.mean()

def launch_optim_nn():
    ### Progress bar 
    progress_text = "Calcul en cours. Veuillez patienter."
    st.session_state['my_bar'] = st.progress(0, text=progress_text)
    ###
    ### Intialisation du temps
    st.session_state['start_time'] = time.time()
    ###

    with st.spinner(st.session_state['spinner_text_modele_optimise']):
        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
        study.optimize(objective_nn, n_trials=st.session_state['nb_trials'], timeout=st.session_state['temps_max'])
        ### Vider la barre de progression
        st.session_state['my_bar'].empty()
        ###
        st.session_state['best_params'] = study.best_params 
        st.session_state['afficher_radar_param_optim'] = True

##################################
### xgboost related functions  ###
##################################

def objective_xgboost(trial):
    # Hyperparamètres à optimiser
    n_estimators = trial.suggest_int('n_estimators', st.session_state['range_nbr_estimateurs'][0], st.session_state['range_nbr_estimateurs'][1])
    max_depth = trial.suggest_int('max_depth', st.session_state['range_max_depth'][0], st.session_state['range_max_depth'][1])
    eta = trial.suggest_float('eta', st.session_state['range_eta'][0], st.session_state['range_eta'][1])
    min_child_weight = trial.suggest_int('min_child_weight', st.session_state['range_min_child_weight'][0]  , st.session_state['range_min_child_weight'][1])

    # Création du modèle
    model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth,
                             eta=eta,
                             min_child_weight=min_child_weight,
                             random_state=123))

    # Score MSE en cross-validation 
    score = cross_val_score(model, st.session_state['X_train_scaled'], st.session_state['y_train_scaled'], cv=3, scoring='neg_mean_squared_error')

    # Score MAPE pour arrêt anticipé
    model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth,
                             eta=eta,
                             min_child_weight=min_child_weight,
                             random_state=123))
    model.fit(st.session_state['X_train_scaled'], st.session_state['y_train_scaled'])
    # Erreur MAPE et RMSE
    if len(st.session_state['outputs']) == 1:
        mape_error, rmse_error = get_MAPE_and_RMSE(st.session_state['y_test_scaled'], st.session_state['X_test_scaled'], st.session_state['scaler_y'], model)
    else:
        mape_error, rmse_error, mape_error_rawvalues, rmse_error_rawvalues = get_MAPE_and_RMSE(st.session_state['y_test_scaled'], st.session_state['X_test_scaled'], st.session_state['scaler_y'], model)

    ### Sauvegarde du meilleur modèle et les scores RMSE et MAPE
    if trial.number == 0:
        st.session_state['model'] = model
        st.session_state['model_MAPE'] = mape_error
        st.session_state['model_RMSE'] = rmse_error
        if len(st.session_state['outputs']) > 1:
            st.session_state['model_MAPE_rawvalues'] = mape_error_rawvalues
            st.session_state['model_RMSE_rawvalues'] = rmse_error_rawvalues
    if rmse_error < st.session_state['model_RMSE']:
        st.session_state['model'] = model
        st.session_state['model_MAPE'] = mape_error
        st.session_state['model_RMSE'] = rmse_error
        if len(st.session_state['outputs']) > 1:
            st.session_state['model_MAPE_rawvalues'] = mape_error_rawvalues
            st.session_state['model_RMSE_rawvalues'] = rmse_error_rawvalues

    ### Arrêt anticipé
    if mape_error < st.session_state['mape_tolerance']:
        trial.study.stop()
        st.success('Arrêt anticipé')

    ### Progress bar 
    temps_chargement = (time.time()-st.session_state['start_time'])/st.session_state['temps_max']
    if temps_chargement > 1:
        temps_chargement = 1
    st.session_state['my_bar'].progress(temps_chargement, text=f'Calcul en cours. Veuillez patienter. Essai n°{trial.number+1}')

    return score.mean()

def launch_optim_xgboost():
    ### Initialisation Progress bar
    progress_text = "Calcul en cours. Veuillez patienter..."
    st.session_state['my_bar'] = st.progress(0, text=progress_text)
    ###
    ### Initialisation du temps
    st.session_state['start_time'] = time.time()
    ###

    with st.spinner(st.session_state['spinner_text_modele_optimise']):
        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
        study.optimize(objective_xgboost, n_trials=st.session_state['nb_trials'], timeout=st.session_state['temps_max'])
        ### vider la barre de progression
        st.session_state['my_bar'].empty()
        ###
        st.session_state['best_params'] = study.best_params
        st.session_state['afficher_radar_param_optim'] = True

#######################################                                                
### Random Forest related functions ###
#######################################

def objective_random_forest(trial):
    ### Hyperparamètres à optimiser
    n_estimators = trial.suggest_int('n_estimators', st.session_state['range_n_estimators'][0], st.session_state['range_n_estimators'][1])
    max_depth = trial.suggest_int('max_depth', st.session_state['range_max_depth'][0], st.session_state['range_max_depth'][1])
    min_samples_split = trial.suggest_int('min_samples_split', st.session_state['range_min_samples_split'][0], st.session_state['range_min_samples_split'][1])
    min_samples_leaf = trial.suggest_int('min_samples_leaf', st.session_state['range_min_samples_leaf'][0], st.session_state['range_min_samples_leaf'][1])

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    random_state=123))

    ### Score MSE en cross-validation
    score = cross_val_score(model, st.session_state['X_train'], st.session_state['y_train'], cv=3, scoring='neg_mean_squared_error')

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    random_state=123))

    model.fit(st.session_state['X_train'], st.session_state['y_train'])

    # Erreur MAPE et RMSE
    if len(st.session_state['outputs']) == 1:
        mape_error, rmse_error = get_MAPE_and_RMSE(st.session_state['y_test'], st.session_state['X_test'], st.session_state['scaler_y'], model)
    else:
        mape_error, rmse_error, mape_error_rawvalues, rmse_error_rawvalues = get_MAPE_and_RMSE(st.session_state['y_test'], 
                                                                                               st.session_state['X_test'], 
                                                                                               st.session_state['scaler_y'], 
                                                                                               model)
        

        

    ### Sauvegarde du meilleur modèle et les scores RMSE et MAPE
    if trial.number == 0:
        st.session_state['model'] = model
        st.session_state['model_MAPE'] = mape_error
        st.session_state['model_RMSE'] = rmse_error
        if len(st.session_state['outputs']) > 1:
            st.session_state['model_MAPE_rawvalues'] = mape_error_rawvalues
            st.session_state['model_RMSE_rawvalues'] = rmse_error_rawvalues
    if rmse_error < st.session_state['model_RMSE']:
        st.session_state['model'] = model
        st.session_state['model_MAPE'] = mape_error
        st.session_state['model_RMSE'] = rmse_error
        if len(st.session_state['outputs']) > 1:
            st.session_state['model_MAPE_rawvalues'] = mape_error_rawvalues
            st.session_state['model_RMSE_rawvalues'] = rmse_error_rawvalues

    ### Arrêt anticipé
    if mape_error < st.session_state['mape_tolerance']:
        trial.study.stop()
        st.success('Arrêt anticipé')

    ### Progress bar 
    temps_chargement = (time.time()-st.session_state['start_time'])/st.session_state['temps_max']
    st.info(temps_chargement)
    if temps_chargement > 1:
        temps_chargement = 1
    st.session_state['my_bar'].progress(temps_chargement, text=f'Calcul en cours. Veuillez patienter. Essai n°{trial.number+1}')

    return score.mean()

def launch_optim_random_forest():
        ### Initialisation Progress bar
        progress_text = "Calcul en cours. Veuillez patienter."
        st.session_state['my_bar'] = st.progress(0, text=progress_text)
        ###
        ### Initialisation du temps
        st.session_state['start_time'] = time.time()
        ###
        
      

        with st.spinner('Optimisation des hyperparamètres...'):
            study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
            study.optimize(objective_random_forest, n_trials=st.session_state['nb_trials'], timeout=st.session_state['temps_max'])
            ### Vider la barre de progression
            st.session_state['my_bar'].empty()
            ###
            st.session_state['best_params'] = study.best_params
            st.session_state['afficher_radar_param_optim'] = True
        return True

### Utilitary functions

def liste_puissance_de_deux(min, max):
    puissance = 1
    while puissance < min:
        puissance *= 2
    puissances = []
    while puissance <= max:
        puissances.append(puissance)
        puissance *= 2

    return puissances

def dizaines(debut, fin):
    dizaines_list = []

    if debut < 10:
        dizaines_list.append(debut)
        debut = 10

    dizaine = (debut // 10) * 10
    if dizaine < debut:
        dizaine += 10

    while dizaine <= fin:
        dizaines_list.append(dizaine)
        dizaine += 10

    return dizaines_list

