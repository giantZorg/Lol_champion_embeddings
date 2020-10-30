###
#
# Embedding erstellen
#
###


###
# Pakete laden

# Zeit
import time

# Pandas
import pandas as pd
import numpy as np

# Mxnet
import mxnet as mx

# MDS
from sklearn import manifold 

# Requests
import requests

# Urllib
import urllib3


# Datenbankverbindung
#import pyodbc
#import sqlalchemy
from sqlalchemy import create_engine

# Plotly
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html

# Matplotlib
from PIL import Image
from io import BytesIO 

from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Fortschrittsanzeige
from tqdm import tqdm

# Logging
import logging


###
# Konstanten

# Region
REGION = 'euw1'
#REGION = 'na1'

# Liste mit zu betrachtenden Champions
CHAMPION_LISTE = ['Aatrox', 'Ahri', 'Akali', 'Alistar', 'Amumu', 'Anivia', 'Annie', 'Aphelios', 'Ashe', 'AurelionSol', 'Azir', 'Bard', 'Blitzcrank', 'Brand', 'Braum', 'Caitlyn', 'Camille', 'Cassiopeia', 'Chogath', 'Corki', 'Darius', 'Diana', 'Draven', 'DrMundo', 'Ekko', 'Elise', 'Evelynn', 'Ezreal', 'Fiddlesticks', 'Fiora', 'Fizz', 'Galio', 'Gangplank', 'Garen', 'Gnar', 'Gragas', 'Graves', 'Hecarim', 'Heimerdinger', 'Illaoi', 'Irelia', 'Ivern', 'Janna', 'JarvanIV', 'Jax', 'Jayce', 'Jhin', 'Jinx', 'Kaisa', 'Kalista', 'Karma', 'Karthus', 'Kassadin', 'Katarina', 'Kayle', 'Kayn', 'Kennen', 'Khazix', 'Kindred', 'Kled', 'KogMaw', 'Leblanc', 'LeeSin', 'Leona', 'Lillia', 'Lissandra', 'Lucian', 'Lulu', 'Lux', 'Malphite', 'Malzahar', 'Maokai', 'MasterYi', 'MissFortune', 'MonkeyKing', 'Mordekaiser', 'Morgana', 'Nami', 'Nasus', 'Nautilus', 'Neeko', 'Nidalee', 'Nocturne', 'Nunu', 'Olaf', 'Orianna', 'Ornn', 'Pantheon', 'Poppy', 'Pyke', 'Qiyana', 'Quinn', 'Rakan', 'Rammus', 'RekSai', 'Renekton', 'Rengar', 'Riven', 'Rumble', 'Ryze', 'Sejuani', 'Senna', 'Sett', 'Shaco', 'Shen', 'Shyvana', 'Singed', 'Sion', 'Sivir', 'Skarner', 'Sona', 'Soraka', 'Swain', 'Sylas', 'Syndra', 'TahmKench', 'Taliyah', 'Talon', 'Taric', 'Teemo', 'Thresh', 'Tristana', 'Trundle', 'Tryndamere', 'TwistedFate', 'Twitch', 'Udyr', 'Urgot', 'Varus', 'Vayne', 'Veigar', 'Velkoz', 'Vi', 'Viktor', 'Vladimir', 'Volibear', 'Warwick', 'Xayah', 'Xerath', 'XinZhao', 'Yasuo', 'Yone', 'Yorick', 'Yuumi', 'Zac', 'Zed', 'Ziggs', 'Zilean', 'Zoe', 'Zyra']


##
# Datenbank
DSN = 'E_LOCAL'

DB_NAME = 'E_LOL_CHAMPS_EMBEDDING'
DB_SCHEMA = 'LOL_CHAMPS_EMBEDDING'

DB_TABELLE_IDS = 'SUMMONER_IDS'
DB_TABELLE_MASTERIES = 'SUMMONER_MASTERIES'


##
# Daten aufarbeiten
GRENZE_OTP = 0.5
MIN_PUNKTE_NOTWENDIG = 10**5

##
# Embeddings
N_EMBEDDING = 16
N_BATCH = 2**12
N_EPOCH = 100

LERN_RATE = 0.005

FAKTOR_RESAMPLING_AUSWAHL = 10
N_ACCOUNTS_PRO_EPOCH = 10000
N_CHAMPS_PRO_ACCOUNT = 15
ACCOUNTS_ZURUECKLEGEN = True

MITTELWERT_TYP = 'geom'     # 'geom', 'harm'
EPS_HARM = 10**-5
EPS_GEOM = 10**-5


SEED = 51
MX_KONTEXT = 'GPU'

##
# Bildausgabe
ICON_ZOOM = 0.03

##
# Sonstiges

# Pfad auf Championbilder
CHAMPION_BILDER_URL = 'http://ddragon.leagueoflegends.com/cdn/10.20.1/img/champion/'

# Logging-Level
LOGGING_LEVEL = 'INFO'



###
# Hilfsfunktionen
#def neuralesNetzDefinieren(nChamps, nDimEmbedding):
#    X = mx.sym.Variable('data')
#    y = mx.sym.Variable('label')
#    
#    symEmb = mx.sym.Embedding(data = X, input_dim = nChamps, output_dim = nDimEmbedding)
#    
#    symEmbChamp1 = mx.sym.slice_axis(symEmb, 1, 0, 1)
#    symEmbChamp2 = mx.sym.slice_axis(symEmb, 1, 1, 2)
#    
#    symEmbReshape1 = mx.sym.reshape(symEmbChamp1, (-1, nDimEmbedding))
#    symEmbReshape2 = mx.sym.reshape(symEmbChamp2, (-1, nDimEmbedding))
#    
##    symEmbNorm1 = mx.sym.broadcast_div(symEmbReshape1, mx.sym.reshape(mx.sym.sqrt(mx.sym.norm(symEmbReshape1, axis = 1)), (-1, 1)))
##    symEmbNorm2 = mx.sym.broadcast_div(symEmbReshape2, mx.sym.reshape(mx.sym.sqrt(mx.sym.norm(symEmbReshape2, axis = 1)), (-1, 1)))
#    
##    symSkalarProdWinkel = mx.sym.sum(symEmbNorm1 * symEmbNorm2, axis = 1, keepdims = True)
#    symSkalarProdWinkel = mx.sym.sum(symEmbReshape1 * symEmbReshape2, axis = 1, keepdims = True)
#    
#    symEbene = mx.sym.FullyConnected(symSkalarProdWinkel, num_hidden = 1)
#    symFuerOutput = mx.sym.Activation(symEbene, act_type = 'tanh')
#    
#    symFehler = mx.sym.LinearRegressionOutput(symFuerOutput, y)
#    
#    return(symFehler)

def neuralesNetzDefinieren(nChamps, nDimEmbedding):
    X = mx.sym.Variable('data')
    y = mx.sym.Variable('label')
    
    symEmb = mx.sym.Embedding(data = X, input_dim = nChamps, output_dim = nDimEmbedding)
    
    symEmbChamp1 = mx.sym.slice_axis(symEmb, 1, 0, 1)
    symEmbChamp2 = mx.sym.slice_axis(symEmb, 1, 1, 2)
    
    symEmbReshape1 = mx.sym.reshape(symEmbChamp1, (-1, nDimEmbedding))
    symEmbReshape2 = mx.sym.reshape(symEmbChamp2, (-1, nDimEmbedding))
    
#    symEmbNorm1 = mx.sym.broadcast_div(symEmbReshape1, mx.sym.reshape(mx.sym.sqrt(mx.sym.norm(symEmbReshape1, axis = 1)), (-1, 1)))
#    symEmbNorm2 = mx.sym.broadcast_div(symEmbReshape2, mx.sym.reshape(mx.sym.sqrt(mx.sym.norm(symEmbReshape2, axis = 1)), (-1, 1)))
    
#    symSkalarProdWinkel = mx.sym.sum(symEmbNorm1 * symEmbNorm2, axis = 1, keepdims = True)
    symSkalarProdWinkel = mx.sym.sum(symEmbReshape1 * symEmbReshape2, axis = 1, keepdims = True)
    
#    symEbene = mx.sym.FullyConnected(symSkalarProdWinkel, num_hidden = 1)
#    symFuerOutput = mx.sym.Activation(symSkalarProdWinkel, act_type = 'softrelu')
    
    symFehler = mx.sym.LinearRegressionOutput(symSkalarProdWinkel, y)
    
    return(symFehler)


###
# Iterator definieren
class embeddingIterator(mx.io.DataIter):
    ###
    # Initialisierung
    def __init__(self, meisterschaftsdatenOhneOTP, batchgroesse = N_BATCH, mittelwertTyp = MITTELWERT_TYP, faktorResamplingAuswahl = FAKTOR_RESAMPLING_AUSWAHL, nAccountsProEpoch = N_ACCOUNTS_PRO_EPOCH, nChampsProAccount = N_CHAMPS_PRO_ACCOUNT, accountsZuruecklegen = ACCOUNTS_ZURUECKLEGEN, EPS_HARM = EPS_HARM, EPS_GEOM = EPS_GEOM):
        # Ablegen
        self.meisterschaftenWerte = meisterschaftsdatenOhneOTP.values
        self.batchgroesse = batchgroesse
        self.mittelwertTyp = mittelwertTyp
        self.faktorResamplingAuswahl = faktorResamplingAuswahl
        self.nAccountsProEpoch = nAccountsProEpoch
        self.nChampsProAccount = nChampsProAccount
        self.accountsZuruecklegen = accountsZuruecklegen
        
        self.EPS_HARM = EPS_HARM
        self.EPS_GEOM = EPS_GEOM
        
        # Für die zufällige Auswahl die Wahrscheinlichkeiten vorbereiten
        self.verhaeltnisse = self.meisterschaftenWerte / self.meisterschaftenWerte.sum(axis = 1).reshape((-1,1))
        
        self.probFuerSampling = self.verhaeltnisse + 1 / (self.verhaeltnisse.shape[1] * faktorResamplingAuswahl)
        self.probFuerSampling = self.probFuerSampling / self.probFuerSampling.sum(axis = 1).reshape((-1,1))
        
        # Initialer Status
        self.reset()

        
    ###
    # Reset
    def reset(self):
        ###
        # Datenmatrix erstellen
        
        # Indizes für Accounts bestimmen
        if self.accountsZuruecklegen:
            maxNAccounts = self.nAccountsProEpoch
        else:
            maxNAccounts = min(self.nAccountsProEpoch, self.meisterschaftenWerte.shape[0])
        
        if maxNAccounts % self.batchgroesse:
            zuesaetzlicheAccounts = self.batchgroesse - maxNAccounts % self.batchgroesse
        else:
            zuesaetzlicheAccounts = 0
        
        indizesAccounts = np.random.choice(self.meisterschaftenWerte.shape[0], maxNAccounts, replace = self.accountsZuruecklegen)
        if zuesaetzlicheAccounts:
            indizesAccounts = np.concatenate((indizesAccounts, np.random.choice(self.meisterschaftenWerte.shape[0], zuesaetzlicheAccounts, replace = True)))
        
        # Indizes verarbeiten
        datenMatrixListe = list()
        for indexAccount in indizesAccounts:
            # Champions herauslesen
            indizesChampions = np.random.choice(self.meisterschaftenWerte.shape[1], self.nChampsProAccount, replace = False, p = self.probFuerSampling[indexAccount,:])
            
            for i in range(0, self.nChampsProAccount):
                for j in range(i+1, self.nChampsProAccount):
                    if self.mittelwertTyp == 'harm':
                        datenMatrixListe.append([indizesChampions[i], indizesChampions[j], 2 * self.verhaeltnisse[indexAccount, indizesChampions[i]] * self.verhaeltnisse[indexAccount, indizesChampions[j]] / (self.verhaeltnisse[indexAccount, indizesChampions[i]] + self.verhaeltnisse[indexAccount, indizesChampions[j]]) if self.verhaeltnisse[indexAccount, indizesChampions[i]] > self.EPS_HARM and self.verhaeltnisse[indexAccount, indizesChampions[j]] > self.EPS_HARM else 0])
#                        datenMatrixListe.append([indizesChampions[i], indizesChampions[j], 2 * (2 * self.verhaeltnisse[indexAccount, indizesChampions[i]] * self.verhaeltnisse[indexAccount, indizesChampions[j]] / (self.verhaeltnisse[indexAccount, indizesChampions[i]] + self.verhaeltnisse[indexAccount, indizesChampions[j]]) if self.verhaeltnisse[indexAccount, indizesChampions[i]] > self.EPS_HARM and self.verhaeltnisse[indexAccount, indizesChampions[j]] > self.EPS_HARM else 0) - 1])
                    elif self.mittelwertTyp == 'geom':
                        datenMatrixListe.append([indizesChampions[i], indizesChampions[j], np.sqrt(self.verhaeltnisse[indexAccount, indizesChampions[i]] * self.verhaeltnisse[indexAccount, indizesChampions[j]]) if self.verhaeltnisse[indexAccount, indizesChampions[i]] > self.EPS_GEOM and self.verhaeltnisse[indexAccount, indizesChampions[j]] > self.EPS_GEOM else 0])
#                        datenMatrixListe.append([indizesChampions[i], indizesChampions[j], 2 * (np.sqrt(self.verhaeltnisse[indexAccount, indizesChampions[i]] * self.verhaeltnisse[indexAccount, indizesChampions[j]]) if self.verhaeltnisse[indexAccount, indizesChampions[i]] > self.EPS_GEOM and self.verhaeltnisse[indexAccount, indizesChampions[j]] > self.EPS_GEOM else 0) - 1])
                    else:
                        raise NotImplementedError('Mittelwerttyp nicht implementiert')
        
        self.datenMatrix = np.array(datenMatrixListe)
        
        self.aktuellerBatch = 0
        self.anzahlBatche = (self.datenMatrix.shape[0] - 1) // self.batchgroesse + 1
    
    
    ###
    # Next
    def next(self):
        if self.aktuellerBatch < self.anzahlBatche:
            # Daten
            daten = mx.nd.array(self.datenMatrix[(self.aktuellerBatch * self.batchgroesse):((self.aktuellerBatch + 1) * self.batchgroesse), 0:2])
            kennzeichnung = mx.nd.array(self.datenMatrix[(self.aktuellerBatch * self.batchgroesse):((self.aktuellerBatch + 1) * self.batchgroesse), 2])
        
            # Batchcounter erhöhen
            self.aktuellerBatch += 1
            
            # Werte ausgeben
            return(mx.io.DataBatch(data = [daten], label = [kennzeichnung], pad = 0))
        else:
            raise(StopIteration)   
    
    
    ###
    # Datenstruktur für die Vorhersagematrizen
    @property
    def provide_data(self):
        return([('data', (self.batchgroesse, 2))])


    ###
    # Datenstruktur für die Kennzeichnungen
    @property
    def provide_label(self):
        return([('label', (self.batchgroesse,))])    


###
# Hauptteil
if __name__ == '__main__':
    ###
    # Logging-Level setzen
    loggingDict = {'DEBUG': 10, 'INFO': 20}
    logging.basicConfig(level = loggingDict[LOGGING_LEVEL])


    # Warnungen zu SSL unterdrücken (da durch Proxy)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    
    ###
    # Datenbankverbindung aufbauen
    logging.debug('Datenbankverbindung wird aufgebaut')
#    dbVerbindung = pyodbc.connect('DSN={}'.format(DSN))
    dbEngine = create_engine('mssql+pyodbc://{}'.format(DSN))
    

    ###
    # ALle vorhandenen Meisterschaftsdaten herunterladen
    
    # Sql-String
    sqlString = 'select {} from {}.{}.{}_{} where summe >= {}'.format(', '.join([x.lower() for x in CHAMPION_LISTE]), DB_NAME, DB_SCHEMA, DB_TABELLE_MASTERIES, REGION.upper(), MIN_PUNKTE_NOTWENDIG)

    # Daten herunterladen
    meisterschaftsdaten = pd.read_sql(sqlString, dbEngine)
    
    
    ###
    # Championbilder herunterladen
    championBilder = list()
    for champion in tqdm(CHAMPION_LISTE, total = len(CHAMPION_LISTE)):
        while True:
            bildAntwort = requests.get('{url}{champion}.png'.format(url = CHAMPION_BILDER_URL, champion = champion))
            if bildAntwort.status_code == 200:
                break
            else:
                time.sleep(1)
        
        championBilder.append(Image.open(BytesIO(bildAntwort.content)))
            
    
#    ###
#    # Einige interessante Übersichten
#    
#    # Verhältnis an Accounts mit 0 Meisterschaftspunkten auf dem entsprechenden Champion
#    verhaeltnisNullPunkte = (np.sum(meisterschaftsdaten == 0) / meisterschaftsdaten.shape[0]).reset_index()
#    verhaeltnisNullPunkte.columns = ['Champion', 'VerhaeltnisNullPunkte']
#    
##    # Plots
##    verhaeltnisNullPunkte['VerhaeltnisNullPunkte'].sort_values().reset_index(drop = True).plot()
#    
##    fig = px.scatter(verhaeltnisNullPunkte, x = verhaeltnisNullPunkte.Champion, y = verhaeltnisNullPunkte.VerhaeltnisNullPunkte)
##    fig.show()
#    
##    import plotly
##    import plotly.graph_objs
##    plotly.offline.plot({
##    "data": [
##        plotly.graph_objs.Scatter(    x=[1, 2, 3, 4],
##        y=[10, 11, 12, 13], mode='markers',
##        marker=dict(
##            size=[40, 60, 80, 100]))],
##    "layout": plotly.graph_objs.Layout(showlegend=False,
##        height=600,
##        width=600,
##    )
##    })
#
#    # Gesamtsumme an Meisterschaftspunkten
#    summeMeisterschaftspunkteProChampion = np.sum(meisterschaftsdaten).reset_index()
#    summeMeisterschaftspunkteProChampion.columns = ['Champion', 'SummeMeisterschaftspunkte']
#    summeMeisterschaftspunkteProChampion['AnteilMeisterschaftspunkte'] = summeMeisterschaftspunkteProChampion['SummeMeisterschaftspunkte'] / sum(summeMeisterschaftspunkteProChampion['SummeMeisterschaftspunkte'])
#
##    summeMeisterschaftspunkteProChampion['AnteilMeisterschaftspunkte'].sort_values().reset_index(drop = True).plot()
#    
#    meisterschaftsdaten.sum(axis = 1).hist()


    ###
    # One-Trick Accounts entfernen
    meisterschaftsdatenOhneOTP = meisterschaftsdaten.loc[meisterschaftsdaten.max(axis = 1) / meisterschaftsdaten.sum(axis = 1) < GRENZE_OTP].reset_index(drop = True)



    ###
    # Neurales Netz definieren
    netzSymb = neuralesNetzDefinieren(meisterschaftsdatenOhneOTP.shape[1], N_EMBEDDING)

    # Iterator
    np.random.seed(SEED)
    mx.random.seed(SEED)
    
    iterator = embeddingIterator(meisterschaftsdatenOhneOTP)
    
    # Neurales Netz initiieren
    if MX_KONTEXT == 'GPU':
        mxKontext = mx.gpu()
    elif MX_KONTEXT == 'CPU':
        mxKontext = mx.cpu()
    else:
        raise AssertionError('MX_KONTEXT hat keinen gültigen Wert')
    
    nnModell = mx.mod.Module(netzSymb, data_names = [x[0] for x in iterator.provide_data], label_names = [x[0] for x in iterator.provide_label], context = mxKontext)

    nnModell.bind(data_shapes = iterator.provide_data, label_shapes = iterator.provide_label)
    nnModell.init_params(mx.initializer.Xavier(rnd_type = 'gaussian', magnitude = 1))
    nnModell.init_optimizer(optimizer = 'adam', optimizer_params = {'learning_rate': LERN_RATE})


    ##
    # Trainieren
#    progressBar = mx.callback.ProgressBar(total = iterator.anzahlBatche)
#    nnModell.fit(iterator, eval_metric = 'mse', num_epoch = N_EPOCH, batch_end_callback = progressBar)
    nnModell.fit(iterator, eval_metric = 'mse', num_epoch = N_EPOCH)


    ###
    # Gewichtungsmatrix auslesen
    gewichtungsMatrix = nnModell.get_params()[0][list(nnModell.get_params()[0].keys())[0]].asnumpy()
    
    # Normierung
    gewichtungsMatrix = gewichtungsMatrix / np.linalg.norm(gewichtungsMatrix, axis = 1).reshape((-1,1))


    ###
    # MDS (2-Dimensional)
    mdsTransformation = manifold.MDS(n_components = 2)
    mdsMatrix = mdsTransformation.fit_transform(gewichtungsMatrix)
    
    mds2dDf = pd.DataFrame(mdsMatrix, columns = ['X1', 'X2'])
    mds2dDf['Champion'] = CHAMPION_LISTE
    
    # Matplotlib
    fig = plt.figure(figsize=(3, 2), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.scatter(mds2dDf['X1'], mds2dDf['X2'], color = 'white')
    ax.axis('off')
    for i in range(0, len(CHAMPION_LISTE)):
        bildBox = OffsetImage(championBilder[i], zoom = ICON_ZOOM)
        ax.add_artist(AnnotationBbox(bildBox, mds2dDf.iloc[i][['X1', 'X2']].values, frameon = False))
    
    fig.savefig("champion_clustering_mda_{region}.png".format(region = REGION), transparent = False, bbox_inches = 'tight', pad_inches = 0, dpi = 1000)
    plt.close()
    
    
    ###
    # TSNE
    tsneTransformation = manifold.TSNE(n_components = 2)
    tsneMatrix = tsneTransformation.fit_transform(gewichtungsMatrix)

    tsne2dDf = pd.DataFrame(tsneMatrix, columns = ['X1', 'X2'])
    tsne2dDf['Champion'] = CHAMPION_LISTE    

    # Matplotlib
    fig = plt.figure(figsize=(3, 2), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.scatter(tsne2dDf['X1'], tsne2dDf['X2'], color = 'white')
    ax.axis('off')
    for i in range(0, len(CHAMPION_LISTE)):
        bildBox = OffsetImage(championBilder[i], zoom = ICON_ZOOM)
        ax.add_artist(AnnotationBbox(bildBox, tsne2dDf.iloc[i][['X1', 'X2']].values, frameon = False))
    
    fig.savefig("champion_clustering_tsne_{region}.png".format(region = REGION), transparent = False, bbox_inches = 'tight', pad_inches = 0, dpi = 1000)
    plt.close()

    
    ###
    # Darstellung Plotly
    if (False):
        fig = px.scatter(mds2dDf, x = 'X1', y = 'X2', hover_data = {'X1': False, 'X2': False, 'Champion': True})
        
        app = dash.Dash()
        app.layout = html.Div([
            dcc.Graph(figure=fig)
        ])
        
        app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter






















