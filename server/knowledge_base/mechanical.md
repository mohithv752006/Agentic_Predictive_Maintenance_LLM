# Mechanical attributes (SMART 7, 10, 188, 240)

Sources:
[1] Wikipedia, "Self-Monitoring, Analysis and Reporting Technology", attribute table. https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology
[2] A. Klein, "What SMART Stats Tell Us About Hard Drives", Backblaze blog, 2016. https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/

## SMART 7: Seek Error Rate
The rate of seek errors of the magnetic heads; a partial failure of the positioning system (for example servo damage or thermal widening) causes seek errors [1]. The raw value has a vendor-specific structure, is often not meaningful as a decimal number, and on some drives grows during normal operation without signifying errors [1]. This system therefore uses the normalized value (higher is better) rather than the raw value.

## SMART 10: Spin Retry Count
Counts retries of spin-start attempts needed to reach full operating speed after a failed first attempt. An increase is a sign of problems in the mechanical subsystem [1]. Healthy drives normally report zero.

## SMART 188: Command Timeout
The count of operations aborted because of a drive timeout; normally this should be zero [1]. Backblaze includes it among its five failure-indicator attributes [2]. Timeouts can also come from cabling, controllers or power, so a burst of timeouts without media errors deserves a closer look rather than an immediate verdict.

## SMART 240: Head Flying Hours
Time spent positioning the drive heads (on some Fujitsu drives this ID reports transfer error counts instead) [1]. It behaves like an age or workload counter rather than an error counter, so it describes exposure rather than damage.
