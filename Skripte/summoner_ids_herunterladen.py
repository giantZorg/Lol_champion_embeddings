###
#
# Skript zum Herunterladen von Beschwörer-IDs
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

# Richtige Kopien
import copy

# Logging
import logging


###
# Konstanten

# Initialer Account, von dem aus andere Accounts gesucht werden
#INITIALER_ACCOUNT = 'giantZorg'     # EUW-1
INITIALER_ACCOUNT = 'Ahnsungtangmyun'

# Region
#REGION = 'euw1'
REGION = 'na1'

# URL-Endpoints
URL_ID_ZU_NAME = 'https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{userName}?api_key={apiKey}'
URL_MATCH_HISTORY = 'https://{region}.api.riotgames.com/lol/match/v4/matchlists/by-account/{accountId}?api_key={apiKey}'
URL_MATCH_INFO = 'https://{region}.api.riotgames.com/lol/match/v4/matches/{matchId}?api_key={apiKey}'

# Wiederholungsversuche falls fehlgeschlagen
URL_RETRIES_TIMES = 10
URL_RETRIES_SLEEP = 5

# Sleep nach einem Download (damit nicht mehr als 100 Downloads pro Minute stattfinden können)
REQUEST_SLEEP = 120/100

# IDs von Bot-Queues (dann wird ein zusätzliicher Check benötigt)
BOT_QUEUE_IDS = set([31, 32, 33, 91, 92, 93, 800, 810, 820, 830, 840, 850, 950, 960])


##
# Datenbank
DSN = 'E_LOCAL'

DB_NAME = 'E_LOL_CHAMPS_EMBEDDING'
DB_SCHEMA = 'LOL_CHAMPS_EMBEDDING'

DB_TABELLE_IDS = 'SUMMONER_IDS'


##
# Sonstiges

# Zeitformat
ZEITFORMAT = '%Y%m%d'
ZEITZONE = 'Europe/Zurich'

# Anzahl Accounts, welche angeschaut werden sollen
N_ACCOUNTS_HERUNTERLADEN = 500

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
    # Zuerst schauen, ob in der DB-Tabelle schon etwas vorhanden ist. Dann dort zufällig einen entnehmen, ansonsten einen initialen Account nehmen
    logging.info('Schon vorhandene Accounts werden heruntergeladen')
    
    # Anzahl schon vorhandene Accounts
    nAccounts = pd.read_sql('select count(*) as N_ACCOUNTS from {}.{}.{}_{}'.format(DB_NAME, DB_SCHEMA, DB_TABELLE_IDS, REGION.upper()), dbEngine)['N_ACCOUNTS'].iloc[0]
    
    if nAccounts:
        logging.debug('Herunterladen des letzten Eintrags')
        alleAccountsDf = pd.read_sql('select summoner_id, summoner_accountid from {}.{}.{}_{}'.format(DB_NAME, DB_SCHEMA, DB_TABELLE_IDS, REGION.upper()), dbEngine)
        
    else:
        ###
        # Initialer Request
        ct = 0
        while True:
            logging.debug('Initialer Account wird heruntergeladen, Versuch {}'.format(ct))
            initialerAccountAntwort = requests.get(URL_ID_ZU_NAME.format(region = REGION, userName = INITIALER_ACCOUNT, apiKey = apiKey), verify = False, proxies = proxies)
            if initialerAccountAntwort.status_code == 200:
                logging.debug('Accountinformation heruntergeladen')
                break
    
            else:
                ct += 1
                time.sleep(URL_RETRIES_SLEEP)
            
            if ct == URL_RETRIES_TIMES:
                logging.error('Initialer Account konnte nicht heruntergeladen werden')
                raise ConnectionError('Initialer Account konnte nicht heruntergeladen werden')
            
        # Json auslesen und als Datenframe umformen
        initialerAccount = json.loads(initialerAccountAntwort.text)
        initialerAccountDf = pd.DataFrame(initialerAccount, index = [0])[['id', 'accountId']]
        
        # Zeit hinzufügen
        initialerAccountDf['zeit'] = int(time.mktime(datetime.today().astimezone(timezone(ZEITZONE)).timetuple()))
        
        # In der Datenbank ablegen
        logging.debug('Initialer Account wird auf die Datenbank geschrieben')
        initialerAccountDf.rename(columns = {'id': 'summoner_id', 'accountId': 'summoner_accountid', 'zeit': 'zeit_erstellt'}).to_sql('{}_{}'.format(DB_TABELLE_IDS, REGION.upper()), dbEngine, schema = '{}.{}'.format(DB_NAME, DB_SCHEMA), index = False, if_exists = 'append')
    
        
        ##
        # Für die Weiterverarbeitung
        alleAccountsDf = initialerAccountDf.drop(columns = 'zeit').rename(columns = {'id': 'summoner_id', 'accountId': 'summoner_accountid'})

    
    ###
    # Seed setzen
    np.random.seed(RANDOM_SEED)
    
    # Flag für schon vorhandene Accounts (weil pandas noch kein upsert kann)
    schonAngeschauteAccounts = set()
    
    for i in tqdm(range(0, N_ACCOUNTS_HERUNTERLADEN), total = N_ACCOUNTS_HERUNTERLADEN, disable = logging.root.level > 20):
        ###
        # Sicherheitscheck, sollte nie geschehen
        if len(schonAngeschauteAccounts) == alleAccountsDf.shape[0]:
            logging.warning('Alle Accounts wurden schon einmal heruntergeladen')
            break
        
        
        ###
        # Auswahl eines Accounts
        while True:
            indexAccount = np.random.choice(alleAccountsDf.shape[0], 1)[0]
            
            # Index noch nicht angeschaut
            if indexAccount not in schonAngeschauteAccounts:
                break
        
        
        ###
        # Letzte Matches für diesen Account herauslesen. Lädt jeweils maximal 100 Matches heraus (entsprechend der API-Konfiguration)
        aktuellerAccount = alleAccountsDf.iloc[indexAccount]

        ct = 0
        while True:
            logging.debug('Matchgeschichte {} wird heruntergeladen, Versuch {}'.format(i, ct))
            matchgeschichteAntwort = requests.get(URL_MATCH_HISTORY.format(region = REGION, accountId = aktuellerAccount['summoner_accountid'], apiKey = apiKey), verify = False, proxies = proxies)
            if matchgeschichteAntwort.status_code == 200:
                logging.debug('Matchgeschichte heruntergeladen')
                break
            
            elif matchgeschichteAntwort.status_code == 429:
                time.sleep(int(matchgeschichteAntwort.headers['Retry-After']) + 1)
                continue
    
            else:
                ct += 1
                print(matchgeschichteAntwort)
                time.sleep(URL_RETRIES_SLEEP)
            
            if ct == URL_RETRIES_TIMES:
                print(matchgeschichteAntwort)
                logging.error('Matchgeschichte konnte nicht heruntergeladen werden')
#                raise ConnectionError('Matchgeschichte konnte nicht heruntergeladen werden')
                break
        
        if matchgeschichteAntwort.status_code != 200:
            continue


        ###
        # Nach dem erfolgreichen Download den Account sperren
        schonAngeschauteAccounts.update(set([indexAccount]))
        
        # Die Matchgeschichte aufarbeiten
        matchgeschichte = json.loads(matchgeschichteAntwort.text)
        if 'matches' in matchgeschichte.keys(): # Eigentlich sollte jeder Account, der so gefunden wurde, auch eine History haben.
            if len(matchgeschichte['matches']): 
                ###
                # Mit der Game-ID die entsprechende Information des Games herauslesen
                for j in tqdm(range(0, len(matchgeschichte['matches'])), total = len(matchgeschichte['matches']), disable = logging.root.level > 20):
                    ###
                    # Matchinformationen herunterladen
                    ct = 0
                    while True:
                        logging.debug('Matchinformation {}, {} wird heruntergeladen, Versuch {}'.format(i, j, ct))
                        matchinformationenAntwort = requests.get(URL_MATCH_INFO.format(region = REGION, matchId = matchgeschichte['matches'][j]['gameId'], apiKey = apiKey), verify = False, proxies = proxies)
                        if matchinformationenAntwort.status_code == 200:
                            logging.debug('Matchinformationen heruntergeladen')
                            break

                        elif matchinformationenAntwort.status_code == 429:
                            time.sleep(int(matchinformationenAntwort.headers['Retry-After']) + 1)
                            continue
                
                        else:
                            ct += 1
                            print(matchinformationenAntwort)
                            time.sleep(URL_RETRIES_SLEEP)
                        
                        if ct == URL_RETRIES_TIMES:
                            print(matchinformationenAntwort)
                            logging.error('Matchinformationen konnten nicht heruntergeladen werden')
#                            raise ConnectionError('Matchinformationen konnten nicht heruntergeladen werden')
                            break
                    
                    if matchinformationenAntwort.status_code != 200:
                        continue

                    ##
                    # Account-IDs auslesen
                    try:
                        matchinformationen = json.loads(matchinformationenAntwort.text)
    
                        # Auslesen
                        if matchinformationen['queueId'] in BOT_QUEUE_IDS:
                            matchAccounts = pd.DataFrame({'summoner_id': [matchinformationen['participantIdentities'][k]['player']['summonerId'] for k in range(0, len(matchinformationen['participantIdentities'])) if 'summonerId' in matchinformationen['participantIdentities'][k]['player'].keys()], 'summoner_accountid': [matchinformationen['participantIdentities'][k]['player']['accountId'] for k in range(0, len(matchinformationen['participantIdentities'])) if 'summonerId' in matchinformationen['participantIdentities'][k]['player'].keys()]})
                        else:
                            matchAccounts = pd.DataFrame({'summoner_id': [matchinformationen['participantIdentities'][k]['player']['summonerId'] for k in range(0, len(matchinformationen['participantIdentities']))], 'summoner_accountid': [matchinformationen['participantIdentities'][k]['player']['accountId'] for k in range(0, len(matchinformationen['participantIdentities']))]})
                    except:
                        continue
                    
                    # Neue Accounts auslesen
                    neueAccounts = copy.deepcopy(matchAccounts.loc[~matchAccounts['summoner_id'].isin(alleAccountsDf['summoner_id'])])
                    
                    ##
                    # Wegschreiben
                    if neueAccounts.shape[0]:
                        # Zeit hinzufügen
                        neueAccounts['zeit_erstellt'] = int(time.mktime(datetime.today().astimezone(timezone(ZEITZONE)).timetuple()))
                        
                        # In der Datenbank ablegen
                        logging.debug('{} neue Accounts werden auf die Datenbank geschrieben'.format(neueAccounts.shape[0]))
                        neueAccounts.to_sql('{}_{}'.format(DB_TABELLE_IDS, REGION.upper()), dbEngine, schema = '{}.{}'.format(DB_NAME, DB_SCHEMA), index = False, if_exists = 'append')
                                        
                        ##
                        # Datenframe erweitern
                        alleAccountsDf = pd.concat([alleAccountsDf, neueAccounts.drop(columns = 'zeit_erstellt')], ignore_index = True)


                    ###
                    # Wartezeit einfügen, damit nicht zuviele Abfragen gemacht werden
                    time.sleep(REQUEST_SLEEP)
                    









