# Thermal attributes (SMART 190, 194)

Sources:
[1] Wikipedia, "Self-Monitoring, Analysis and Reporting Technology", attribute table and field-study summary. https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology
[2] E. Pinheiro, W.-D. Weber, L. A. Barroso, "Failure Trends in a Large Disk Drive Population", USENIX FAST 2007. https://www.usenix.org/legacy/event/fast07/tech/full_papers/pinheiro/pinheiro.pdf

## SMART 194: Temperature
Reports the device temperature if a sensor is fitted; the lowest byte of the raw value contains the temperature in degrees Celsius [1]. Some vendors pack additional data into the higher bytes, so very large raw values should be read as encoded rather than literal temperatures.

## SMART 190: Airflow Temperature / Temperature Difference
On many drives the normalized value equals 100 minus the temperature in degrees Celsius, so lower normalized values mean hotter drives; some older drives report the raw temperature here instead, or the temperature minus 50 [1].

## How much temperature matters
The Google field study found little correlation between increased temperature and failure, and no correlation with usage level ([2], as summarized in [1]). In this system thermal evidence is weighted by the calibrated model rather than by fixed thresholds; a sustained rise relative to the drive's own baseline, or a short spike, is more informative than the absolute level.
