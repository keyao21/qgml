@ECHO OFF 
echo Starting tests
pause 

::echo DOUBLE_GYRE_ACTUAL
::echo python .\generate_velocity_fields.py --demo DOUBLE_GYRE_ACTUAL
::python .\generate_velocity_fields.py --demo DOUBLE_GYRE_ACTUAL

cd src
FOR %%A IN ( TESTING ) DO (
    ECHO python generate_streamfunction_values.py --demo %%A
    python generate_streamfunction_values.py --demo %%A
    ECHO python generate_velocity_fields.py --demo %%A
    python generate_velocity_fields.py --demo %%A  
)