# Limits of SMART-based prediction and vendor caveats

Sources:
[1] Wikipedia, "Self-Monitoring, Analysis and Reporting Technology". https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology
[2] A. Klein, "What SMART Stats Tell Us About Hard Drives", Backblaze blog, 2016. https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/
[3] E. Pinheiro, W.-D. Weber, L. A. Barroso, "Failure Trends in a Large Disk Drive Population", USENIX FAST 2007. https://www.usenix.org/legacy/event/fast07/tech/full_papers/pinheiro/pinheiro.pdf

## Many failures give no SMART warning
In the Google study, 56% of failed drives had no count in the four strongest SMART signals (scan errors, reallocation count, offline reallocation, probational count), and 36% failed with no SMART error at all apart from temperature ([3], as summarized in [1]). Backblaze reports that 23.3% of failed drives showed no warning in its five tracked attributes [2]. A low predicted risk is therefore not a guarantee of health.

## Non-zero is not the same as failing
4.2% of operational Backblaze drives had at least one of the five tracked attributes above zero [2]. Single non-zero values need context: the rate of change and the combination of attributes matter more than the presence of a count [2].

## Vendor-specific raw values
Several raw values have vendor-specific structure: seek error rate raw values are often not meaningful as decimal numbers [1], the temperature raw value keeps the temperature in its lowest byte [1], and power-on time may be counted in hours, minutes or seconds [1]. Comparisons are only meaningful within one drive model.
