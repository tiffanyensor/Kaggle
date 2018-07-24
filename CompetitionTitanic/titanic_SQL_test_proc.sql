-- Function: public.bld_wrk_titanic_test()

-- DROP FUNCTION public.bld_wrk_titanic_test();

CREATE OR REPLACE FUNCTION public.bld_wrk_titanic_test()
  RETURNS void AS
$BODY$ 
BEGIN

-- ======================
-- CREATE RAW TABLE
-- ======================

CREATE TABLE RAW_titanic_test
(
Rownumber SERIAL,
PassengerID VARCHAR(1000),
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

DROP TABLE IF EXISTS WRK_Titanic_test;

-- ======================
-- CREATE WRK TABLE
-- ======================

CREATE TABLE WRK_Titanic_test
(
Rownumber SERIAL,
PassengerID VARCHAR(100),
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

TRUNCATE TABLE WRK_Titanic_test;

-- ======================
-- INSERT DATA
-- ======================

INSERT INTO WRK_Titanic_test
(PassengerID, PClass, Name, Sex, Age, SibSp, Parch,
Ticket, Fare, Cabin, Embarked
)
SELECT
PassengerID,
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
FROM RAW_Titanic_test;

-- ======================
-- FILTERS
-- ======================

SELECT * FROM WRK_Titanic_test
WHERE age IS NULL;
-- 86 rows affected

SELECT * FROM WRK_Titanic_test
WHERE fare IS NULL;
-- 1 row affected

-- Total rows with missing data: 87

      
END;      
$BODY$
  LANGUAGE plpgsql VOLATILE
  COST 100;
ALTER FUNCTION public.bld_wrk_titanic_test()
  OWNER TO postgres;
