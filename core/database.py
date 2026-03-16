from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config import POSTGRES_CONFIG_TRAIN, POSTGRES_CONFIG_PROD

database_train = POSTGRES_CONFIG_TRAIN["dbname"]
user_train = POSTGRES_CONFIG_TRAIN["user"]
password_train = POSTGRES_CONFIG_TRAIN["password"]
host_train = POSTGRES_CONFIG_TRAIN["host"]
port_train = POSTGRES_CONFIG_TRAIN["port"]

database_prod = POSTGRES_CONFIG_PROD["dbname"]
user_prod = POSTGRES_CONFIG_PROD["user"]
password_prod = POSTGRES_CONFIG_PROD["password"]
host_prod = POSTGRES_CONFIG_PROD["host"]
port_prod = POSTGRES_CONFIG_PROD["port"]

# URLs de connexion
SQLALCHEMY_DATABASE_URL_TRAIN = "postgresql://{user}:{password}@{host}/{database}".format(user=user_train, password=password_train, host=host_train, database=database_train)
SQLALCHEMY_DATABASE_URL_PROD = "postgresql://{user}:{password}@{host}/{database}".format(user=user_prod, password=password_prod, host=host_prod, database=database_prod)

engine_train = create_engine(SQLALCHEMY_DATABASE_URL_TRAIN)
engine_prod = create_engine(SQLALCHEMY_DATABASE_URL_PROD)

SessionLocalTrain = sessionmaker(autocommit=False, autoflush=False, bind=engine_train)
SessionLocalProd = sessionmaker(autocommit=False, autoflush=False, bind=engine_prod)

# Les modèles destinés à la base 'train' hériteront de BaseTrain
BaseTrain = declarative_base()

# Les modèles destinés à la base 'prod' hériteront de BaseProd
BaseProd = declarative_base()

def get_db_train():
    db = SessionLocalTrain()
    try:
        yield db
    finally:
        db.close()

def get_db_prod():
    db = SessionLocalProd()
    try:
        yield db
    finally:
        db.close()
