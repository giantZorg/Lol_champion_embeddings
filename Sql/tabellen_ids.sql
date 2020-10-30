/*
Tabellendefinition SUMMONER_IDS_EUW1
*/

drop table if exists LOL_CHAMPS_EMBEDDING.SUMMONER_IDS_EUW1;
drop table if exists LOL_CHAMPS_EMBEDDING.SUMMONER_IDS_NA1;


-- Beschwörer-IDs und wann geladen
create table LOL_CHAMPS_EMBEDDING.SUMMONER_IDS_EUW1 (summoner_id NVARCHAR(63), summoner_accountid NVARCHAR(56), zeit_erstellt BIGINT, primary key (summoner_id, summoner_accountid));
create table LOL_CHAMPS_EMBEDDING.SUMMONER_IDS_NA1 (summoner_id NVARCHAR(63), summoner_accountid NVARCHAR(56), zeit_erstellt BIGINT, primary key (summoner_id, summoner_accountid));


