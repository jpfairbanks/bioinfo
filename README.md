bioinfo
=======

Scripts for processing bioinformatics data

## plotter.py
Reads from some CSV files and makes a bunch of residual plots
The most important step is resolving the different indices.

### Data Format
The data should be in a comma separated values file. In the following format.
It can be exported from Excel.
experimental_times, experimental_data,blank,time(secs),time(mins),model_data

The filenames are significant as the information for titling the plots is extracted
from the filename.

`data.Cathep.Cathep.substrate.csv`
where there can be as many Cathep entries as you want.

### Invoking the Program
Go to your nearest terminal and type
`python -i plotter data.*`

The data.* is a wildcard that passes in all files that start with data. to the program.

If you want to see residual plots you can use the command flag -r like 
`python -i plotter -r data.*`
For verbose progress statements you can use --verbose
`python -i plotter --verbose data.*`
These can be combined.
