import streamlit as st
from ut import *

st.title('Bac à sable Réseau de neurones')

#############################################
###            Onglets                    ###
#############################################

tab1, tab2, tab3 = st.tabs(["Entrainement sur une range d\'hyperparamètres","Entrainement d\'un modèle dédié", "Téléchargement du modèle"])

##############################################
###             Tab 1                      ###
##############################################

with tab1:
    st.subheader('modele par output')
    # dictionnaire key : output, value : model

    st.subheader('Choix du modèle')
    model = st.selectbox('Choix du modèle', ['Random Forest', 'Neural Network', 'XGBoost', 'CatBoost'])
    st.session_state['afficher_radar_param_optim'] = False

    st.subheader('Choix des hyperparamètres')
     
    ###################################
    ##            XGBoost            ##
    ###################################

    if model == 'XGBoost':

        col_1, col_2 = st.columns([1,1])
        
        with col_1:
            st.session_state['slider_range_nbr_estimateurs'] = st.slider('Nombre d\'estimateurs', value=[50, 200], step=10, min_value=20, max_value=500)
            st.session_state['slider_range_max_depth'] = st.slider('Profondeur maximale', value=[3, 10], step=1, min_value=1, max_value=15)
            st.session_state['slider_range_eta'] = st.slider('Taux d\'apprentissage', value=[0.01, 0.2], step=0.1, min_value=0.01, max_value=0.5)
            st.session_state['slider_range_min_child_weight'] = st.slider('Poids minimal des feuilles', value=[1, 3], step=1, min_value=1, max_value=10)
            
            st.session_state['nb_trials'] = st.number_input('Nombre d\'essais e combinaisons', min_value=1, max_value=1000, value=20)
            
            

            st.session_state['categories'] = ['nombre d\'estimateurs', 'profondeur maximale', 'taux d\'apprentissage', 'poids minimal des feuilles']

            if st.button('Valider les choix'):
                launch_optim_xgboost()
                best_param = st.session_state['best_params']
                best_nb_estimators = best_param['n_estimators']
                best_max_depth = best_param['max_depth']
                best_eta = best_param['eta']
                best_min_child_weight = best_param['min_child_weight']
                ###### Faire une fonction
                best_model = xgb.XGBRegressor(n_estimators=best_nb_estimators,
                                             max_depth=best_max_depth,
                                             eta=best_eta,
                                             min_child_weight=best_min_child_weight,
                                             random_state=123)
                ######
                best_model.fit(st.session_state['X_train'], st.session_state['y_train'])
                st.session_state['model'] = best_model
                st.session_state['type_model'] = 'XGBoost'

        with col_2:
            get_radar_xgboost_slider()
            if st.session_state['afficher_radar_param_optim']:
                get_radar_xgboost_optim()
                st.write(st.session_state['best_params'])

    ####################################
    ##        Neural Network          ##
    ####################################

    elif model == 'Neural Network':
        col_1, col_2 = st.columns([1,1])

        with col_1:
            
            ## Saisie des hyperparamètres de l'utilisateur ##
            st.session_state['slider_range_first_layer'] = st.select_slider('Premiere couche', options=[2,4,8,16,32,64,128], value=(8,64))
            st.session_state['slider_range_second_layer'] = st.select_slider('Seconde couche', options=[2,4,8,16,32,64,128], value=(2,32))
            st.session_state['slider_range_batch_size'] = st.select_slider('Taille des batches', options=[1,10, 20, 30, 40, 50,60,70,80,100], value=(10, 50))
            st.session_state['nb_epoch'] = st.number_input('Nombre d\'itération parmi les données d\entrainement', min_value=50, max_value=1000, value=500)
    

            st.session_state['nb_trials'] = st.number_input('Nombre d\'essais', min_value=1, max_value=1000, value=10)

            st.session_state['categories'] = ['nombre de neurones première couche', 'nombre de neurones seconde couche', 'taille de batch'] 

            if st.button('Valider les choix'):
                launch_optim_nn()

               
                best_first_layer = st.session_state['best_params']['first_layer']
                best_second_layer = st.session_state['best_params']['second_layer']
                best_batch_size = st.session_state['best_params']['batch_size']
                best_model = MLPRegressor(hidden_layer_sizes=[best_first_layer, best_second_layer],
                                            batch_size=best_batch_size,
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
                                            n_iter_no_change=50)
                best_model.fit(st.session_state['X_scaled'], st.session_state['y'])
                st.session_state['model'] = best_model
                st.session_state['type_model'] = 'Neural network'

                    
                    
                    
        with col_2:
            get_radar_nn_slider()
            if st.session_state['afficher_radar_param_optim']:
                get_radar_nn_optim()
                st.write(st.session_state['best_params'])
        
    ####################################
    ##        Random Forest           ##
    ####################################

    elif model == 'Random Forest':
        col_1, col_2 = st.columns([1,1])

        with col_1:
            st.session_state['slider_range_n_estimators'] = st.slider('Number of estimators', value=[50,200], min_value=10, max_value=500)
            st.session_state['slider_range_max_depth'] = st.slider('Max depth', value=[3, 10], min_value=1, max_value=50)
            st.session_state['slider_range_min_samples_split'] = st.slider('Min samples split', value=[2,5], min_value=2, max_value=10)
            st.session_state['slider_range_min_samples_leaf'] = st.slider('Min samples leaf', value=[1,5], min_value=1, max_value=10)

            st.session_state["nb_trials"] = st.number_input('Number of trials', min_value=1, max_value=1000, value=20)

            st.session_state['categories'] = ['number of estimators', 'max depth', 'min samples split', 'min samples leaf']

            if st.button('Validate choices'):
                launch_optim_random_forest()
                best_param = st.session_state['best_params']
                best_n_estimators = best_param['n_estimators']
                best_max_depth = best_param['max_depth']
                best_min_samples_split = best_param['min_samples_split']
                best_min_samples_leaf = best_param['min_samples_leaf']
                best_model = RandomForestRegressor(n_estimators=best_n_estimators,
                                                    max_depth=best_max_depth,
                                                    min_samples_split=best_min_samples_split,
                                                    min_samples_leaf=best_min_samples_leaf,
                                                    random_state=123)
                best_model.fit(st.session_state['X_train'], st.session_state['y_train'])
                st.session_state['model'] = best_model
                st.session_state['type_model'] = 'Random Forest'

        with col_2:
            get_radar_random_forest_slider()
            if st.session_state['afficher_radar_param_optim']:
                get_radar_random_forest_optim()
                st.write(st.session_state['best_params'])

    ####################################
    ##        CatBoost                ##
    ####################################

    elif model == 'CatBoost':
        col_1, col_2 = st.columns([1,1])

        with col_1:
            st.session_state['slider_range_n_iterations'] = st.slider('Number of iterations', value=[100,500], min_value=50, max_value=1000)
            st.session_state['slider_range_learning_rate'] = st.slider('Learning rate', value=[0.01, 0.1], min_value=0.001, max_value=0.5)
            st.session_state['slider_range_depth'] = st.slider('Depth', value=[3, 10], min_value=1, max_value=15)
            st.session_state['slider_range_subsample'] = st.slider('Subsample', value=[0.5, 0.8], min_value=0.05, max_value=1.)

            st.session_state['nb_trials'] = st.number_input('Number of trials', min_value=1, max_value=1000, value=20)

            st.session_state['categories'] = ['number of iterations', 'learning rate', 'depth', 'subsample']

            if st.button('Validate choices'):
                launch_optim_catboost()
                best_param = st.session_state['best_params']
                best_n_iterations = best_param['n_iterations']
                best_learning_rate = best_param['learning_rate']
                best_depth = best_param['depth']
                best_subsample = best_param['subsample']
                best_model = cb.CatBoostRegressor(iterations=best_n_iterations,
                                               learning_rate=best_learning_rate,
                                               depth=best_depth,
                                               subsample=best_subsample,
                                               random_state=123)
                best_model.fit(st.session_state['X_train'], st.session_state['y_train'])
                st.session_state['model'] = best_model
                st.session_state['type_model'] = 'CatBoost'

                

        with col_2 :
            get_radar_catboost_slider()
            if st.session_state['afficher_radar_param_optim']:
                get_radar_catboost_optim()
                st.write(st.session_state['best_params'])


##############################################
###             Tab 2                      ###
##############################################


with tab2:
    col1, col2 = st.columns([2,2])
    with col1:
        
        if True:

            with st.container(border=True):
                st.subheader('Nom du modèle :')
                nom = st.text_input('Nom du modèle', 'modele_personnalisé')
            ########################################
            ###       Architecture du modèle     ###
            ########################################
    
            with st.container(border=True):
        
                st.subheader('Architecture du réseau de neurones')
                nb_hidden_player = st.number_input('Nombre de couches cachées', min_value=1, max_value=10, value = 2)
                nb_neurons = []
                nb_activation = []
                for _ in range(nb_hidden_player):
                    nb_neurons.append(st.number_input(f'Nombre de neurones pour la couche {_+1}', min_value=1, max_value=500, value = 64, key= _))
                    nb_activation.append(st.selectbox(f'Fonction d\'activation pour la couche {_+1}', [ 'leaky_relu','relu', 'tanh', 'sigmoid', 'selu', 'elu' ]))

        


            ########################################
            ###       Compilation                ###
            ########################################
            with st.container(border=True):
                st.subheader('Hyperparamètres de compilation')

                # Optimizer
                optimizer = st.selectbox('Optimizer', ['adam', 'rmsprop', 'sgd'], placeholder='adam') 
                # Loss
                loss = st.selectbox('Loss', ['mse', 'mae', 'huber'], placeholder='mse')
                # Metrics
                metrics = st.multiselect('Metrics', ['mse', 'mae', 'huber', 'mape'], default = ['mape', 'mae'])

            #########################################
            ###       Prétraitement des données   ###
            #########################################

            with st.container(border=True):
                st.subheader('Prétraitement des données et paramètres d\'entrainement')

                # Scaler 
                scaler = st.selectbox('Scaler', ['StandardScaler', 'MinMaxScaler', 'RobustScaler'], placeholder='StandardScaler')

                # Train test split
                test_size = st.number_input('Taille de l\'échantillon de test (en %)', min_value=5, max_value=95, value = 20)
                valid_size = st.number_input('Taille de l\'échantillon de validation (en %)', min_value=5, max_value=95, value = 20)
                test_size = test_size / 100
                valid_size = valid_size / 100

                # nb epochs
                epochs = st.number_input('Nombre d\'itération lors de l\'entraînement', min_value=1, max_value=10000, value = 500)
            
            #########################################
            ###       Callbacks                   ###
            #########################################

            with st.container(border=True):
                st.subheader('call backs ')
                # Early stopping
                st.write('suivant les metriques choisies, le callback early stopping sera mis en place ave les seuils :')
                if st.checkbox('Early stopping', value=True):
                    metric_earlystopping = st.selectbox('Metric early stopping', metrics)
                    threshold_earlystop = st.number_input(f' - {metric_earlystopping} :', value=10.)

                # Model checkpoint
                # Reduce LR on plateau ( On SGD and RMSprop )
                # Cyclical lr 
                cyclic = st.checkbox('Cyclical learning rate')
                st.write(cyclic)
            

                # Afficher en temps réel l'entrainement du modèle
                display_graph = st.checkbox('Afficher en temps réel l\'entrainement du modèle')
                st.write(display_graph)

            ##########################################
            ###       Entrainement du modèle       ###
            ##########################################
            if 'trained_bool' not in st.session_state:
                st.session_state.trained_bool = False
            if st.button('Valider la saiie et démarrer l\'entraînement'):
                model = create_customized_model(st.session_state.X.shape[1], nb_neurons, nb_activation, st.session_state.y.shape[1])
                st.info('Modèle créé')
                model = compile_customized_model(model, optimizer, loss, metrics)
                st.info('Modèle compilé')
                 
                st.session_state['personalized_X_train'],st.session_state['personalized_X_test'],st.session_state['personalized_Y_train'] ,st.session_state['personalized_Y_test'] ,st.session_state['personalized_X_scaler'],st.session_state['personalized_Y_scaler'] = preprocess_data(st.session_state['X'],
                                                                            st.session_state['y'], 
                                                                            scaler, 
                                                                            test_size)
                st.info('Données prétraitées')
                st.info({st.session_state['personalized_X_train'].shape})
                st.info(st.session_state['personalized_X_train'][:5])
                #callbacks = create_customized_callbacks(metrics, {metric_earlystopping: threshold_earlystop}, cyclic, display_graph)
                #st.info('Callbacks créés')
                with st.spinner('Entraînement du modèle en cours...'):
                    st.session_state.model, history = train_customized_model(model, st.session_state['personalized_X_train'], st.session_state['personalized_Y_train'], epochs, 32, valid_size)
                st.info('Modèle entraîné')
                err = evaluate_customized_model(st.session_state.model, st.session_state['personalized_X_test'], st.session_state['personalized_Y_test'])
                st.info(f'Erreur de prédiction : {err}')
                st.info(model.metrics_names)
                st.session_state.trained_bool = True


        else:
            st.info('Veuillez charger un jeu de données')

    with col2:
        ########################################
        ###        Architecture              ###
        ########################################

        st.write('Un réseau de neurones avec 2 couches cachées peut généralement être capable d\'approximer n\'importe quelle fonction continue. \n')
        st.write('Il est donc possible de tester différentes architectures pour trouver celle qui convient le mieux à votre problème. \n')

        ########################################
        ###        Compilation               ###
        ########################################

        st.write('L\'optimiseur est responsable de la mise à jour des poids du réseau de neurones. \n')

        st.write('La fonction de perte est une mesure de la qualité de la prédiction du modèle. \n')
        st.latex(r'L(y, f(x)) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(x_i))')

        st.write('Les métriques sont des indicateurs de performance du modèle. \n')

        ########################################
        ###        Prétraitement             ###
        ########################################

        st.write('Le prétraitement des données est une étape cruciale pour l\'entraînement d\'un modèle de réseau de neurones. \n')
        st.write('Il est important de normaliser les données pour faciliter la convergence du modèle. \n')
        st.write('Généralement, la normalisation gaussienne (StandardScaler) est la plus efficace pour les données numériques. \n')

        st.write('Il est également important de diviser les données en un ensemble d\'entraînement et un ensemble de test. \n')
        st.write('L\'ensemble de test est utilisé pour évaluer la performance du modèle. \n')
        st.write('Le nombre d\'itérations (epochs) est le nombre de fois que le modèle va parcourir l\'ensemble d\'entraînement. \n')
        st.write('A noter que le modèle peut arrêter l\'entraînement avant la fin des epochs si la performance ne s\'améliore plus. \n,C\'est une variable à l\'appréciation de l\'utilisateur. \n')

        ########################################
        ###        Callbacks                 ###
        ########################################

        st.write('Les callbacks sont des fonctions qui sont appelées à des moments précis lors de l\'entraînement du modèle. \n')
        st.write('Ils permettent de contrôler le comportement du modèle pendant l\'entraînement. \n')
        st.write('Par exemple, le callback EarlyStopping arrête l\'entraînement si la performance du modèle ne s\'améliore plus. \n')



##############################################
###             Tab 3                      ###
##############################################

with tab3:
    if st.session_state.trained_bool:
        st.write('Téléchargement du modèle')
        download_model_and_scalers(st.session_state.model, st.session_state.personalized_X_scaler, st.session_state.personalized_Y_scaler, nom)
        
        
    