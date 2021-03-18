# X opencv

### Rules:
- Please package the code you wrote each time into a separate python package according to the date.
- Each package must be named after "d" + date(MMDD).
- Each package must have an only top-level entry point function named "run".

- if the package has multiple subpackages, we need to follow these rules below:
    - the top-level entry point function of each package should be "run" + PACKAGE_ENTRY_POINT_SERIAL_NUMBER. (ex: run_2)
    - add import statements into the __init__.py file of the package.

### Run:
1. you can provide project name in command line arguments, for example `python3 main.py 0228`, '0228' means February twenty-eight.
   
2. Or provide date after running this program (it will ask you for this).

3. if the package has multiple subpackages, we can add a sub package serial number after the command line argument, for example `python3 main.py 0228 4`, '4' means the 4th sub package. This rule can be also used after running this program.