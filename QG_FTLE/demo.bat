@ECHO OFF 
echo Starting demo file
pause 

::echo DOUBLE_GYRE_ACTUAL
::echo python .\generate_velocity_fields.py --demo DOUBLE_GYRE_ACTUAL
::python .\generate_velocity_fields.py --demo DOUBLE_GYRE_ACTUAL

cd src
FOR %%A IN ( DOUBLE_GYRE_ACTUAL DOUBLE_GYRE_EST QUASIGEO_ACTUAL QUASIGEO_EST ) DO (
    ECHO python generate_velocity_fields.py --demo %%A
    python generate_velocity_fields.py --demo %%A  
)