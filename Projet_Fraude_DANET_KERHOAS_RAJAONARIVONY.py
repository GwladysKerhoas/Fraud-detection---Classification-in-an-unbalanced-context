# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:28:55 2022

@author: chloe
"""

import pandas as pd
import calendar
import h2o
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from numpy import where
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from h2o.estimators import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OKMeansEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
from h2o.estimators import H2ODeepLearningEstimator
from sklearn.neighbors import KNeighborsClassifier
from h2o.estimators import H2ONaiveBayesEstimator
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator


#importation du jeu de données
df = pd.read_csv('guillaume.txt', header=0, sep=";", decimal=",", skiprows=[1956361])
#pour trouver l'index de la ligne d'entête au milieu des données
#df[df['DateTransaction']=='DateTransaction']==True 

#affichage des premières lignes
print(df.head())

#affichage des dernières lignes
print(df.tail())

#dimension du jeu de données
print(df.shape)

#enumération des colonnes
print(df.columns)

#liste des variables et leur type
print(df.dtypes)

#informations sur les données
print(df.info())

#description des données
print(df.describe())



# ------------------------------------------------------------------
#   Pré-traitement des données 
# ------------------------------------------------------------------

#vérification des valeurs manquantes
df.count()

#conversion de la date de transaction en datetime des df
df['DateTransaction'] = pd.to_datetime(df['DateTransaction'])

#création de nouvelles colonnes pour l'analyse des variables
df['Mois']=df['DateTransaction'].dt.month
df['Mois']=df['Mois'].apply(lambda x: calendar.month_abbr[x]).astype('str') 
Jour_Semaine={0:'Lundi', 1:'Mardi', 2:'Mercredi', 3:'Jeudi', 4:'Vendredi', 5:'Samedi', 6:'Dimanche'}
df['Jour']=df['DateTransaction'].dt.dayofweek.map(Jour_Semaine)
df['Jour']=df['Jour'].astype('str') 

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Analyse Univariée :
#   Procédures d'analyse de la distribution des données 
# ------------------------------------------------------------------

#séparaison des transactions acceptées et refusées
good=df.loc[df['FlagImpaye'] == 0,:]
fraud=df.loc[df['FlagImpaye'] ==1,:]
good=good.reset_index(drop = True)
fraud=fraud.reset_index(drop = True)

#pourcentage de fraudes
nb_good = len(good.index)
nb_fraud = len(fraud.index)
print(nb_fraud/(nb_fraud+nb_good)*100)

#visualisation du montant en fonction du refus ou de l'acceptation du chèque
good["Montant"].plot(kind='kde', xlim=(0,1500))
fraud["Montant"].plot(kind='kde', xlim=(0,1500))
plt.legend(['transactions acceptées','transactions refusées'])
plt.show()

#nombre de refus de chèques selon le jour de la semaine
f, ax = plt.subplots(figsize=(15, 8))
sns.lineplot(x=df['Mois'], y=df['FlagImpaye'],data=df, hue='Jour', ax=ax).set_title('Nombre de transaction refusée selon le jour de la semaine')

#montant dépensé par mois selon le jour de la semaine
f, ax = plt.subplots(figsize=(15, 8))
sns.lineplot(x=df['Mois'], y=df['Montant'], data=df, hue='Jour', ax=ax).set_title('Montant dépensé par mois selon le jour de la semaine')

#matrice de corrélation
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(ax=ax,data=df.corr(), annot = True)

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Préparation de l'échantillonage
# ------------------------------------------------------------------

#partionnement train et test
df_train=df.loc[df["DateTransaction"].between('2017-02-01', '2017-08-31')]
df_test=df.loc[df["DateTransaction"].between('2017-09-01', '2017-11-30')]

#séparation X et y
ytrain = df_train['FlagImpaye']
Xtrain= df_train.loc[:, ['Montant','CodeDecision', 'VerifianceCPT1' ,'VerifianceCPT2', 'VerifianceCPT3', 
'D2CB','ScoringFP1','ScoringFP2','ScoringFP3' ,'TauxImpNb_RB' ,'TauxImpNB_CPM','EcartNumCheq', 'NbrMagasin3J',
'DiffDateTr1' ,'DiffDateTr2','DiffDateTr3','CA3TRetMtt','CA3TR']]

ytest = df_test['FlagImpaye']
Xtest= df_test.loc[:, ['Montant','CodeDecision', 'VerifianceCPT1' ,'VerifianceCPT2', 'VerifianceCPT3', 
'D2CB','ScoringFP1','ScoringFP2','ScoringFP3' ,'TauxImpNb_RB' ,'TauxImpNB_CPM','EcartNumCheq', 'NbrMagasin3J',
'DiffDateTr1' ,'DiffDateTr2','DiffDateTr3','CA3TRetMtt','CA3TR']]

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Normalisation des données 
# ------------------------------------------------------------------

ss = StandardScaler()
Xtraincr = pd.DataFrame(ss.fit_transform(Xtrain),columns = Xtrain.columns,index = Xtrain.index)
Xtestcr = pd.DataFrame(ss.fit_transform(Xtest),columns = Xtest.columns,index = Xtest.index)
train_cr=Xtraincr
train_cr["FlagImpaye"] = ytrain
test_cr=Xtestcr
test_cr["FlagImpaye"] = ytest

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Echantillonnage déséquillibré 
#   Suréchantillonner la classe minoritaire : stratégie SMOTE 
# ------------------------------------------------------------------

#SMOTE
#https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

#distribution originale 
print(Counter(ytrain))

#définition du pipeline
over = SMOTE(sampling_strategy=0.1) #sur-échantillonage de la classe minoritaire
under = RandomUnderSampler(sampling_strategy=0.2) #sous-échnatillonage de la classe majoritaire
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

#transformation du dataset
XtrainSmote, ytrainSmote = pipeline.fit_resample(Xtraincr, ytrain)

#nouvelle distribution
counter = Counter(ytrainSmote)
print(counter)

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Initialisation de H2o et création des dataframes adaptés 
# ------------------------------------------------------------------

# Initialisation de H2o
h2o.init()

# Transformation des données en H2oFrame
trainSmote = XtrainSmote
trainSmote["FlagImpaye"] = ytrainSmote
trainh2oSmote = h2o.H2OFrame(trainSmote, column_types={"FlagImpaye": "categorical"})
testh2o_cr = h2o.H2OFrame(test_cr, column_types={"FlagImpaye": "categorical"})

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Méthode d'apprentissage non supervisée : K-means 
# ------------------------------------------------------------------

#construction du modèle
fraud_kmeans = H2OKMeansEstimator(k=2,
                                 estimate_k=True,
                                 standardize=True,
                                 seed=12345)

fraud_kmeans.train(x=trainh2oSmote.columns[:-1],
                  training_frame=trainh2oSmote,
                  validation_frame=testh2o_cr)

#évaluation des performances
perform_kmeans = fraud_kmeans.model_performance()
print(perform_kmeans)

#prédictions sur les données test
pred_K_means = fraud_kmeans.predict(testh2o_cr).as_data_frame()
print(pred_K_means)

#matrice de confusion
crosstab_K_means = pd.crosstab(testh2o_cr.as_data_frame()["FlagImpaye"],pred_K_means['predict'])
print(crosstab_K_means)

#auc
auc_K_means = roc_auc_score(testh2o_cr.as_data_frame()["FlagImpaye"],pred_K_means['predict'])
print(auc_K_means)
#0.49269206947431365

#F1-score
f1score_K_means = (2*crosstab_K_means[1][1])/(2*crosstab_K_means[1][1]+crosstab_K_means[1][0]+crosstab_K_means[0][1])
print(f1score_K_means)
#0.01609887466584668

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Méthode d'apprentissage supervisée : Regression Logistique 
# ------------------------------------------------------------------

#construction du modèle
fraud_reg_log = H2OGeneralizedLinearEstimator(seed=1234, 
                                              family= "binomial", 
                                              lambda_ = 0, 
                                              standardize=True,
                                              keep_cross_validation_predictions=True,
                                              nfolds=5)

#application du modèle sur les données d'apprentissage
fraud_reg_log.train(x=trainh2oSmote.columns[:-1],
                    y='FlagImpaye', 
                    training_frame=trainh2oSmote)

#affichage des coefficients sur les données standardisées
print(fraud_reg_log.coef_norm())

#évaluation des performances
perform_reg_log = fraud_reg_log.model_performance()
print(perform_reg_log)

#prédictions sur les données test
pred_Reg_log = fraud_reg_log.predict(testh2o_cr).as_data_frame()
print(pred_Reg_log)

#matrice de confusion
crosstab_Reg_log = pd.crosstab(testh2o_cr.as_data_frame()["FlagImpaye"],pred_Reg_log['predict'])
print(crosstab_Reg_log)

#auc
auc_Reg_log = roc_auc_score(testh2o_cr.as_data_frame()["FlagImpaye"],pred_Reg_log['predict'])
print(auc_Reg_log)
#0.8638988328530954

#F1-score
f1score_Reg_log = (2*crosstab_Reg_log[1][1])/(2*crosstab_Reg_log[1][1]+crosstab_Reg_log[1][0]+crosstab_Reg_log[0][1])
print(f1score_Reg_log)
#0.45287995101963924 / 0.4064715203068968

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Méthode d'apprentissage supervisée : K-Nearest-Neighbors 
# ------------------------------------------------------------------

#réduction du nombre de données dans l'échantillon d'apprentissage pour réduire les temps de calcul
#partionnement train et test
df_train_knn = df.loc[df["DateTransaction"].between('2017-06-01', '2017-08-31')]
df_test_knn = df.loc[df["DateTransaction"].between('2017-09-01', '2017-10-01')]
#séparation X et y
ytrain_knn = df_train['FlagImpaye']
Xtrain_knn = df_train.loc[:, ['Montant','CodeDecision', 'VerifianceCPT1' ,'VerifianceCPT2', 'VerifianceCPT3', 
'D2CB','ScoringFP1','ScoringFP2','ScoringFP3' ,'TauxImpNb_RB' ,'TauxImpNB_CPM','EcartNumCheq', 'NbrMagasin3J',
'DiffDateTr1' ,'DiffDateTr2','DiffDateTr3','CA3TRetMtt','CA3TR']]
ytest_knn = df_test['FlagImpaye']
Xtest_knn = df_test.loc[:, ['Montant','CodeDecision', 'VerifianceCPT1' ,'VerifianceCPT2', 'VerifianceCPT3', 
'D2CB','ScoringFP1','ScoringFP2','ScoringFP3' ,'TauxImpNb_RB' ,'TauxImpNB_CPM','EcartNumCheq', 'NbrMagasin3J',
'DiffDateTr1' ,'DiffDateTr2','DiffDateTr3','CA3TRetMtt','CA3TR']]
#normalisation des données
ss = StandardScaler()
Xtraincr_knn = pd.DataFrame(ss.fit_transform(Xtrain_knn),columns = Xtrain_knn.columns,index = Xtrain_knn.index)
Xtestcr_knn = pd.DataFrame(ss.fit_transform(Xtest_knn),columns = Xtest_knn.columns,index = Xtest_knn.index)
train_cr_knn = Xtraincr_knn
train_cr_knn["FlagImpaye"] = ytrain_knn
test_cr_knn = Xtestcr_knn
test_cr_knn["FlagImpaye"] = ytest_knn
#smote
over = SMOTE(sampling_strategy=0.1) #sur-échantillonage de la classe minoritaire
under = RandomUnderSampler(sampling_strategy=0.2) #sous-échnatillonage de la classe majoritaire
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
XtrainSmote_knn, ytrainSmote_knn = pipeline.fit_resample(Xtraincr_knn, ytrain_knn)


#modèle
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(XtrainSmote_knn, ytrainSmote_knn)

#prédictions sur les données test
ypred_knn = classifier.predict(Xtestcr_knn)

#matrice de confusion
crosstab_knn = pd.crosstab(ytest_knn,ypred_knn)
print(crosstab_knn)

#auc
auc_knn = roc_auc_score(ytest_knn,ypred_knn)
print(auc_knn)

#F1-score
fmesure_knn = f1_score(ytest_knn,ypred_knn)
print(fmesure_knn)

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Méthode d'apprentissage supervisée : Random Forest 
# ------------------------------------------------------------------

# Instanciation
rf = H2ORandomForestEstimator(seed=1234,ntrees=20,max_depth=200,min_rows=100,nfolds=2,keep_cross_validation_predictions = True)

# Apprentissage
rf.train(x=trainh2oSmote.columns[:-1],y="FlagImpaye",training_frame=trainh2oSmote)

predRf = rf.predict(testh2o_cr).as_data_frame()

crosstaboverRf = pd.crosstab(testh2o_cr.as_data_frame()["FlagImpaye"],predRf['predict'])
print(crosstaboverRf)

rf_auc = roc_auc_score(testh2o_cr.as_data_frame()["FlagImpaye"],predRf['predict'])
print(rf_auc)
# 0.7013391496716729

#F1-Score
print(f1_score(testh2o_cr.as_data_frame()["FlagImpaye"],predRf.predict,pos_label=1))
# 0.030468054814023846


# GridSearch (trop long)

# GBM hyperparameters
gbm_params1 = {'learn_rate': [0.01, 0.1],
                'max_depth': [5,8,10],
                'sample_rate': [0.8, 1.0],
                'col_sample_rate': [0.2, 0.5, 1.0]}

# Train and validate a cartesian grid of GBMs
gbm_grid1 = H2OGridSearch(model=H2OGradientBoostingEstimator,
                          grid_id='gbm_grid1',
                          hyper_params=gbm_params1)
gbm_grid1.train(x=trainh2oSmote.columns[:-1], y="FlagImpaye",
                training_frame=trainh2oSmote,
                ntrees=200,
                seed=1234)

# Get the grid results, sorted by F1
gbm_gridperf1 = gbm_grid1.get_grid(sort_by='f1', decreasing=True)
gbm_gridperf1

# Grab the top GBM model, chosen by F1
best_gbm1 = gbm_gridperf1.models[0]
print(best_gbm1)

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Méthode d'apprentissage supervisée : Gradient Boosting
# ------------------------------------------------------------------

# Build and train the model:
pros_gbm = H2OGradientBoostingEstimator(nfolds=2,
                                        seed=1234,
                                        keep_cross_validation_predictions = True)

pros_gbm.train(x=trainh2oSmote.columns[:-1],y="FlagImpaye",training_frame=trainh2oSmote)

# Eval performance:
print(pros_gbm.model_performance())
print(pros_gbm.summary())

# Generate predictions on a test set (if necessary):
predGbm = pros_gbm.predict(testh2o_cr).as_data_frame()

# Extract feature interactions:
feature_interactions = pros_gbm.feature_interaction()
print(feature_interactions)

#matrice de confusion
crosstabGbm = pd.crosstab(testh2o_cr.as_data_frame()["FlagImpaye"],predGbm['predict'])
print(crosstabGbm)

#auc
aucGbm = roc_auc_score(testh2o_cr.as_data_frame()["FlagImpaye"],predGbm['predict'])
print(aucGbm)
# 0.609894340875416

#F1-Score
fscoreGbm = f1_score(testh2o_cr.as_data_frame()["FlagImpaye"],predGbm.predict)
print(fscoreGbm)
# 0.02390586570925311


# Utilisation d'un échantillon de validation

#subdivision
trainLearning, trainValidation = trainh2oSmote.split_frame(ratios=[0.70],seed=1111)
#dimensions - learning set
trainLearning.shape #(1623564, 19)
#dimensions - validation set
trainValidation.shape #(695508, 19)

#instanciation
gbBis=H2OGradientBoostingEstimator(seed=1111,ntrees=50,max_depth=5, nfolds=5,
stopping_metric="log_loss",stopping_rounds=3,stopping_tolerance=1e-3)
#apprentissage
gbBis.train(x=trainh2oSmote.columns[:-1], y="FlagImpaye", training_frame=trainLearning, validation_frame=trainValidation)

#évolution
pros_gbm.plot()
gbBis.plot()

#résumé
gbBis.summary()


# Correction

gbBis=H2OGradientBoostingEstimator(seed=1111,ntrees=50,max_depth=5)
#apprentissage
gbBis.train(x=trainh2oSmote.columns[:-1], y="FlagImpaye", training_frame=trainLearning, validation_frame=trainValidation)

#prediction - de nouveau voir le seuil d'affectation
predGbBis = gbBis.predict(testh2o_cr).as_data_frame()
print(predGbBis.head(10))

#F1-Score
fscoreGbBis = f1_score(testh2o_cr.as_data_frame()["FlagImpaye"],predGbBis.predict)
print(fscoreGbBis)
#0.07094384829470332

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Méthode d'apprentissage supervisée : Neural Networks 
# ------------------------------------------------------------------

#construction du modèle
fraud_neural = H2ODeepLearningEstimator(seed=1234,
                                        epochs=1250,
                                        standardize=True,
                                        hidden=[8],
                                        activation="Tanh",
                                        distribution="bernoulli", 
                                        export_weights_and_biases = True,
                                        keep_cross_validation_predictions=True,
                                        nfolds=5)

#application du modèle sur les données d'apprentissage
fraud_neural.train(x=trainh2oSmote.columns[:-1], 
                   y="FlagImpaye",
                   training_frame=trainh2oSmote)

# Visualisation du réseau de neurones
fraud_neural.summary()

#évaluation des performances
perform_neural = fraud_neural.model_performance()
print(perform_neural)

#prédictions sur les données test
pred_neural = fraud_neural.predict(testh2o_cr).as_data_frame()
print(pred_neural)

#matrice de confusion
crosstab_neural = pd.crosstab(testh2o_cr.as_data_frame()["FlagImpaye"],pred_neural['predict'])
print(crosstab_neural)

#auc
auc_neural = roc_auc_score(testh2o_cr.as_data_frame()["FlagImpaye"],pred_neural['predict'])
print(auc_neural)
#0.856845464185434 / 0.8699687785702678

#F1-score
f1score_neural = (2*crosstab_neural[1][1])/(2*crosstab_neural[1][1]+crosstab_neural[1][0]+crosstab_neural[0][1])
print(f1score_neural)
#0.5508798775822494 / 0.3759054517727793 / 0.6563695416817034

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Méthode d'apprentissage supervisée : Naive Bayes
# ------------------------------------------------------------------

#instanciation
fraud_nb = H2ONaiveBayesEstimator(seed=1234)

#apprentissage
fraud_nb.train(x=trainh2oSmote.columns[:-1], 
               y="FlagImpaye",
               training_frame=trainh2oSmote)

#affichage
fraud_nb.show()

#évaluation des performances
perform_nb = fraud_nb.model_performance()
print(perform_nb)

#prediction sur les données test
pred_nb = fraud_nb.predict(testh2o_cr).as_data_frame()
print(pred_nb.head())

#matrice de confusion
crosstab_nb = pd.crosstab(testh2o_cr.as_data_frame()["FlagImpaye"],pred_nb['predict'])
print(crosstab_nb)

#auc
auc_nb = roc_auc_score(testh2o_cr.as_data_frame()["FlagImpaye"],pred_nb['predict'])
print(auc_nb)
#0.8541745970408772

#F1-score
f1score_nb = (2*crosstab_nb[1][1])/(2*crosstab_nb[1][1]+crosstab_nb[1][0]+crosstab_nb[0][1])
print(f1score_nb)
#0.16608322324966976

#ou
print(f1_score(testh2o_cr.as_data_frame()["FlagImpaye"],pred_nb.predict,pos_label=1))
#0.16608322324966976

# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Bagging des deux meilleurs modèles et des deux moins bons
# ------------------------------------------------------------------

# Régression logistique et réseau de neurones
Bag_Reg_NN = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial",
                                       base_models=[fraud_reg_log, fraud_neural])

Bag_Reg_NN.train(x=trainh2oSmote.columns[:-1],
               y="FlagImpaye",
               training_frame=trainh2oSmote)


#prédictions sur les données test
pred_Bag_Reg_NN = Bag_Reg_NN.predict(testh2o_cr).as_data_frame()
print(pred_Bag_Reg_NN)

#matrice de confusion
crosstab_Bag_Reg_NN = pd.crosstab(testh2o_cr.as_data_frame()["FlagImpaye"],pred_Bag_Reg_NN['predict'])
print(crosstab_Bag_Reg_NN)

#auc
auc_Bag_Reg_NN = roc_auc_score(testh2o_cr.as_data_frame()["FlagImpaye"],pred_Bag_Reg_NN['predict'])
print(auc_Bag_Reg_NN)
#0.8607061474698035

#F1-score
f1score_Bag_Reg_NN = (2*crosstab_Bag_Reg_NN[1][1])/(2*crosstab_Bag_Reg_NN[1][1]+crosstab_Bag_Reg_NN[1][0]+crosstab_Bag_Reg_NN[0][1])
print(f1score_Bag_Reg_NN)
#0.5186238582289558 / 0.632390391630532

# ------------------------------------------------------------------

# Random Forest et Gradient Boosting
Bag_RF_GradB = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial_bis",
                                       base_models=[rf, pros_gbm])

Bag_RF_GradB.train(x=trainh2oSmote.columns[:-1],
               y="FlagImpaye",
               training_frame=trainh2oSmote)


#prédictions sur les données test
pred_Bag_RF_GradB = Bag_RF_GradB.predict(testh2o_cr).as_data_frame()
print(pred_Bag_RF_GradB)

#matrice de confusion
crosstab_Bag_RF_GradB = pd.crosstab(testh2o_cr.as_data_frame()["FlagImpaye"],pred_Bag_RF_GradB['predict'])
print(crosstab_Bag_RF_GradB)

#auc
auc_Bag_RF_GradB = roc_auc_score(testh2o_cr.as_data_frame()["FlagImpaye"],pred_Bag_RF_GradB['predict'])
print(auc_Bag_RF_GradB)
#0.7099576526557458

#F1-score
f1score_Bag_RF_GradB = (2*crosstab_Bag_RF_GradB[1][1])/(2*crosstab_Bag_RF_GradB[1][1]+crosstab_Bag_RF_GradB[1][0]+crosstab_Bag_RF_GradB[0][1])
print(f1score_Bag_RF_GradB)
#0.031356680285287986

# ------------------------------------------------------------------



# ------------------------------------------------------------------
#   Courbe ROC de comparaison des modèles supervisés
# ------------------------------------------------------------------    

#modeles = [rf, rf_under, rf_over, rf_smoteunder]
modeles = {'fraud_reg_log': fraud_reg_log, "fraud_neural": fraud_neural, 'gbBis':gbBis, 'rf': rf, 'fraud_nb':fraud_nb, 'Bag_Reg_NN':Bag_Reg_NN, 'Bag_RF_GradB' : Bag_RF_GradB}

for modelestr, modele in modeles.items():
    globals()["out_"+modelestr] = modele.model_performance(testh2o_cr)
    globals()["fpr_"+modelestr]= globals()["out_"+modelestr].fprs
    globals()["tpr_"+modelestr]= globals()["out_"+modelestr].tprs
    globals()["aire_" +modelestr]=auc(globals()["fpr_"+modelestr],globals()["tpr_"+modelestr])
    

plt.figure(figsize=(10,8))
lw = 2
plt.plot(fpr_fraud_reg_log, tpr_fraud_reg_log, color='red', lw=lw, label='ROC curve Reg_Log' % aire_fraud_reg_log)
plt.plot(fpr_fraud_neural, tpr_fraud_neural, color='green', lw=lw, label='ROC curve Neural Networks' % aire_fraud_neural)
plt.plot(fpr_rf, tpr_rf, color='blue', lw=lw, label='ROC curve Random Forest' % aire_rf)
plt.plot(fpr_fraud_nb, tpr_fraud_nb, color='orange', lw=lw, label='ROC curve Naive Bayes' % aire_fraud_nb)
plt.plot(fpr_gbBis, tpr_gbBis, color='purple', lw=lw, label='ROC curve Gradient Boosting' % aire_gbBis)
plt.plot(fpr_Bag_Reg_NN, tpr_Bag_Reg_NN, color='pink', lw=lw, label='ROC curve Bagging Reg Log / NN' % aire_Bag_Reg_NN)
plt.plot(fpr_Bag_RF_GradB, tpr_Bag_RF_GradB, color='yellow', lw=lw, label='ROC curve Bagging Random Forest / Gradient Boosting' % aire_Bag_RF_GradB)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.show()

# ------------------------------------------------------------------   


















