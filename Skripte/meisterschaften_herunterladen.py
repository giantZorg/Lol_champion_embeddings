###
#
# Skript zum Herunterladen der Championmeisterschaftspunkte für gegebene Beschwörer-IDs
#
###


###
# Pakete laden

# OS, Sys
import os

# Zeit
import time
from datetime import datetime
from pytz import timezone

# Json
import json

# Requests
import requests

# Urllib
import urllib3

# Pandas
import pandas as pd
import numpy as np

# Datenbankverbindung
#import pyodbc
#import sqlalchemy
from sqlalchemy import create_engine

# Fortschrittsanzeige
from tqdm import tqdm

# Logging
import logging


###
# Konstanten

# Region
#REGION = 'euw1'
REGION = 'na1'

# URL-Endpoints
URL_CHAMPION_INFO = 'http://ddragon.leagueoflegends.com/cdn/10.20.1/data/en_US/champion.json'
URL_MASTERY_POINTS = 'https://{region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/{summonerId}?api_key={apiKey}'

# Liste mit zu betrachtenden Champions
CHAMPION_LISTE = ['Aatrox', 'Ahri', 'Akali', 'Alistar', 'Amumu', 'Anivia', 'Annie', 'Aphelios', 'Ashe', 'AurelionSol', 'Azir', 'Bard', 'Blitzcrank', 'Brand', 'Braum', 'Caitlyn', 'Camille', 'Cassiopeia', 'Chogath', 'Corki', 'Darius', 'Diana', 'Draven', 'DrMundo', 'Ekko', 'Elise', 'Evelynn', 'Ezreal', 'Fiddlesticks', 'Fiora', 'Fizz', 'Galio', 'Gangplank', 'Garen', 'Gnar', 'Gragas', 'Graves', 'Hecarim', 'Heimerdinger', 'Illaoi', 'Irelia', 'Ivern', 'Janna', 'JarvanIV', 'Jax', 'Jayce', 'Jhin', 'Jinx', 'Kaisa', 'Kalista', 'Karma', 'Karthus', 'Kassadin', 'Katarina', 'Kayle', 'Kayn', 'Kennen', 'Khazix', 'Kindred', 'Kled', 'KogMaw', 'Leblanc', 'LeeSin', 'Leona', 'Lillia', 'Lissandra', 'Lucian', 'Lulu', 'Lux', 'Malphite', 'Malzahar', 'Maokai', 'MasterYi', 'MissFortune', 'MonkeyKing', 'Mordekaiser', 'Morgana', 'Nami', 'Nasus', 'Nautilus', 'Neeko', 'Nidalee', 'Nocturne', 'Nunu', 'Olaf', 'Orianna', 'Ornn', 'Pantheon', 'Poppy', 'Pyke', 'Qiyana', 'Quinn', 'Rakan', 'Rammus', 'RekSai', 'Renekton', 'Rengar', 'Riven', 'Rumble', 'Ryze', 'Sejuani', 'Senna', 'Sett', 'Shaco', 'Shen', 'Shyvana', 'Singed', 'Sion', 'Sivir', 'Skarner', 'Sona', 'Soraka', 'Swain', 'Sylas', 'Syndra', 'TahmKench', 'Taliyah', 'Talon', 'Taric', 'Teemo', 'Thresh', 'Tristana', 'Trundle', 'Tryndamere', 'TwistedFate', 'Twitch', 'Udyr', 'Urgot', 'Varus', 'Vayne', 'Veigar', 'Velkoz', 'Vi', 'Viktor', 'Vladimir', 'Volibear', 'Warwick', 'Xayah', 'Xerath', 'XinZhao', 'Yasuo', 'Yone', 'Yorick', 'Yuumi', 'Zac', 'Zed', 'Ziggs', 'Zilean', 'Zoe', 'Zyra']

# Wiederholungsversuche falls fehlgeschlagen
URL_RETRIES_TIMES = 10
URL_RETRIES_SLEEP = 5

# Sleep nach einem Download (damit nicht mehr als 100 Downloads pro Minute stattfinden können)
REQUEST_SLEEP = 120/100


##
# Datenbank
DSN = 'E_LOCAL'

DB_NAME = 'E_LOL_CHAMPS_EMBEDDING'
DB_SCHEMA = 'LOL_CHAMPS_EMBEDDING'

DB_TABELLE_IDS = 'SUMMONER_IDS'
DB_TABELLE_MASTERIES = 'SUMMONER_MASTERIES'


##
# Sonstiges

# Zeitformat
ZEITFORMAT = '%Y%m%d'
ZEITZONE = 'Europe/Zurich'

# Anzahl Accounts, welche angeschaut werden sollen
N_MASTERIES_HERUNTERLADEN = 19325

# Zufallsseed
RANDOM_SEED = 51

# Logging-Level
LOGGING_LEVEL = 'INFO'



###
# Hauptteil
if __name__ == '__main__':
    ###
    # Logging-Level setzen
    loggingDict = {'DEBUG': 10, 'INFO': 20}
    logging.basicConfig(level = loggingDict[LOGGING_LEVEL])
    
    
    ###
    # API-Key laden
    apiKey = os.environ.get('RIOT_API_KEY_EMBEDDING')
    
    # Einfache Checks
    if apiKey is None:
        logging.error('Kein API-Key in der ENV-Variable RIOT_API_KEY gefunden')
        raise AssertionError('Kein API-Key in der ENV-Variable RIOT_API_KEY gefunden')
    elif len(apiKey) != 42:
        logging.error('Kein API-Key mit gütliger Länge in der ENV-Variable RIOT_API_KEY gefunden')
        raise AssertionError('Kein API-Key mit gütliger Länge in der ENV-Variable RIOT_API_KEY gefunden')
    else:
        logging.debug('API-Key geladen')

    # Warnungen zu SSL unterdrücken (da durch Proxy)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    
    ###
    # Proxy-Informationen laden
    logging.debug('Proxy-Informationen werden geladen')
    proxies = {'http': os.environ.get('http_proxy'), 'https': os.environ.get('https_proxy')}
    
    # Falls keine Proxyinformationen gefunden wurden
    if proxies['http'] is None or proxies['https'] is None:
        proxies = None
        
    
    ###
    # Datenbankverbindung aufbauen
    logging.debug('Datenbankverbindung wird aufgebaut')
#    dbVerbindung = pyodbc.connect('DSN={}'.format(DSN))
    dbEngine = create_engine('mssql+pyodbc://{}'.format(DSN))
    

    ###
    # Zuerst einmal noch die anzuschauenden Champions definieren (damit keine allzu neuen Champions ohne Mastery Points vorhanden sind) und die Champion-IDs laden
    ct = 0
    while True:
        logging.debug('Championinformationen werden heruntergeladen, Versuch {}'.format(ct))
        championInformationenAntwort = requests.get(URL_CHAMPION_INFO, verify = False, proxies = proxies)
        if championInformationenAntwort.status_code == 200:
            logging.debug('Championinformationen heruntergeladen')
            break

        else:
            ct += 1
            time.sleep(URL_RETRIES_SLEEP)
        
        if ct == URL_RETRIES_TIMES:
            logging.error('Championinformationen konnten nicht heruntergeladen werden')
            raise ConnectionError('Championinformationen konnten nicht heruntergeladen werden')

    ##
    # Championliste erstellen
    championInformationen = json.loads(championInformationenAntwort.text)
    championDict = {championInformationen['data'][champion]['id']: championInformationen['data'][champion]['key'] for champion in championInformationen['data'].keys() if champion in CHAMPION_LISTE}

    championIdsVorhanden = set(map(int, championDict.values()))
    championIdsReihenfolgeDb = [int(championDict[champion]) for champion in CHAMPION_LISTE]
    
    championListeDb = list(map(lambda x: x.lower(), CHAMPION_LISTE))


    ###
    # Herunterladen der Summoner-IDs
    logging.info('Summoner-IDs werden heruntergeladen')
    sqlString = 'select summoner_id from {}.{}.{}_{} except select summoner_id from {}.{}.{}_{}'.format(DB_NAME, DB_SCHEMA, DB_TABELLE_IDS, REGION.upper(), DB_NAME, DB_SCHEMA, DB_TABELLE_MASTERIES, REGION.upper())
    summonerIds = pd.read_sql(sqlString, dbEngine)
    
    if summonerIds.shape[0] == 0:
        logging.error('Keine Summoner-IDs heruntergeladen')
        raise AssertionError('Keine Summoner-IDs heruntergeladen')


    ###
    # Indexvektor für die Zufallsauswahl
    np.random.seed(RANDOM_SEED)
    indexVektor = np.random.permutation(summonerIds.shape[0])


    ###
    # Champion-Masteries herunterladen und auswerten
    for i in tqdm(range(0, min(N_MASTERIES_HERUNTERLADEN, summonerIds.shape[0])), total = min(N_MASTERIES_HERUNTERLADEN, summonerIds.shape[0]), disable = logging.root.level > 20):
        ###
        # Herunterladen
        ct = 0
        while True:
            logging.debug('Championmeisterschaften {} werden heruntergeladen, Versuch {}'.format(i, ct))
            championMeisterschaftenAntworten = requests.get(URL_MASTERY_POINTS.format(region = REGION, summonerId = summonerIds['summoner_id'].iloc[indexVektor[i]], apiKey = apiKey), verify = False, proxies = proxies)
            if championMeisterschaftenAntworten.status_code == 200:
                logging.debug('Championmeisterschaften heruntergeladen')
                break
            
            elif championMeisterschaftenAntworten.status_code == 429:
                time.sleep(int(championMeisterschaftenAntworten.headers['Retry-After']) + 1)
                continue
    
            else:
                ct += 1
                time.sleep(URL_RETRIES_SLEEP)
            
            if ct == URL_RETRIES_TIMES:
                print(championMeisterschaftenAntworten)
                logging.error('Championmeisterschaften konnten nicht heruntergeladen werden')
#                raise ConnectionError('Championmeisterschaften konnten nicht heruntergeladen werden')
        
        if championMeisterschaftenAntworten.status_code != 200:
            continue

        ###
        # Als Dict umformen und auswerten
        championMeisterschaften = json.loads(championMeisterschaftenAntworten.text)
        championMeisterschaftenAccount = {champion['championId']: champion['championPoints'] for champion in championMeisterschaften if champion['championId'] in championIdsVorhanden}
        
        # Champions ohne Materies einfügen
        fehlendeChampions = championIdsVorhanden - set(championMeisterschaftenAccount.keys())
        
        if len(fehlendeChampions):
            for championId in fehlendeChampions:
                championMeisterschaftenAccount[championId] = 0
        
        # Richtig sortieren
        championMeisterschaftenDf = pd.DataFrame(championMeisterschaftenAccount, index = [0])[championIdsReihenfolgeDb]
        
        # Erweitern und danach die Namen richtig setzen
        championMeisterschaftenDf = pd.concat([pd.DataFrame({'summoner_id': summonerIds['summoner_id'].iloc[indexVektor[i]], 'summe': sum(championMeisterschaftenDf.iloc[0]), 'zeit_erstellt': int(time.mktime(datetime.today().astimezone(timezone(ZEITZONE)).timetuple()))}, index = [0]), championMeisterschaftenDf], axis = 1)
        championMeisterschaftenDf.columns = ['summoner_id', 'summe', 'zeit_erstellt'] + championListeDb


        ###
        # Auf der Datenbank ablegen
        championMeisterschaftenDf.to_sql('{}_{}'.format(DB_TABELLE_MASTERIES, REGION.upper()), dbEngine, schema = '{}.{}'.format(DB_NAME, DB_SCHEMA), index = False, if_exists = 'append')

        # Sleep
        time.sleep(REQUEST_SLEEP)





