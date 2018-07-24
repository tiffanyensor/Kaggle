-- Function: public.bld_wrk_titanic_train()

CREATE OR REPLACE FUNCTION public.bld_wrk_titanic_train()
  RETURNS void AS
$BODY$ 
BEGIN

-- ======================
-- CREATE RAW TABLE
-- ======================

CREATE TABLE RAW_titanic_train
(
Rownumber SERIAL,
PassengerID VARCHAR(1000),
Survived VARCHAR(1000),
PClass VARCHAR(1000),
Name VARCHAR(1000),
Sex VARCHAR(1000),
Age VARCHAR(1000),
SibSp VARCHAR(1000),
Parch VARCHAR(1000),
Ticket VARCHAR(1000),
Fare VARCHAR(1000),
Cabin VARCHAR(1000),
Embarked VARCHAR(1000)
);


-- ======================
-- DROP TABLE
-- ======================

DROP TABLE IF EXISTS WRK_Titanic_train;

-- ======================
-- CREATE WRK TABLE
-- ======================

CREATE TABLE WRK_Titanic_train
(
Rownumber SERIAL,
PassengerID VARCHAR(100),
Survived VARCHAR(1),
PClass INT,
Name VARCHAR(1000),
Sex VARCHAR(6),
Age FLOAT,
SibSp INT,
Parch INT,
Ticket VARCHAR(1000),
Fare FLOAT,
Cabin VARCHAR(100),
Embarked VARCHAR(1)
);

-- ======================
-- TRUNCATE TABLE
-- ======================

TRUNCATE TABLE WRK_Titanic_train;

-- ======================
-- INSERT DATA
-- ======================

INSERT INTO WRK_Titanic_train
(PassengerID, Survived, PClass, Name, Sex, Age, SibSp, Parch,
Ticket, Fare, Cabin, Embarked
)
SELECT
PassengerID,
CAST(Survived as INT),
CAST(PClass as INT),
Name,
Sex,
CAST(Age as FLOAT),
CAST(SibSp as INT),
CAST(Parch as INT),
Ticket,
CAST(Fare as FLOAT),
Cabin,
Embarked
FROM RAW_Titanic_train;

-- ======================
-- FILTERS
-- ======================

SELECT * FROM WRK_Titanic_train
WHERE age IS NULL;
-- 177 rows affected

SELECT * FROM WRK_Titanic_train
WHERE embarked IS NULL;
-- 2 rows affected

-- Total rows with missing data: 179

      
END;      
$BODY$
  LANGUAGE plpgsql VOLATILE
  COST 100;
ALTER FUNCTION public.bld_wrk_titanic_train()
  OWNER TO postgres;
