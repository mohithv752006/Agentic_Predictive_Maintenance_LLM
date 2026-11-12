# Media health attributes (SMART 5, 184, 187, 197, 198)

Sources:
[1] Wikipedia, "Self-Monitoring, Analysis and Reporting Technology", attribute table. https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology
[2] A. Klein, "What SMART Stats Tell Us About Hard Drives", Backblaze blog, 2016. https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/
[3] E. Pinheiro, W.-D. Weber, L. A. Barroso, "Failure Trends in a Large Disk Drive Population", USENIX FAST 2007. https://www.usenix.org/legacy/event/fast07/tech/full_papers/pinheiro/pinheiro.pdf

## SMART 5: Reallocated Sectors Count
The raw value counts bad sectors that were found and remapped to spare sectors; a higher value means the drive has had to reallocate more sectors [1]. A drive with any reallocations is significantly more likely to fail in the following months [1]. Backblaze investigates a drive when the raw value is greater than zero, but a small isolated value (for example two remapped sectors) means little on its own; it matters in combination with other signals and with how fast it grows [2].

## SMART 197: Current Pending Sector Count
Counts "unstable" sectors waiting to be remapped after unrecoverable read errors. If a pending sector is later read successfully it is remapped and the count decreases [1]. Unlike the other attributes Backblaze tracks, SMART 197 is not cumulative, so it can go down as well as up [2]. A rising pending count together with a rising reallocated count suggests the surface is deteriorating [2].

## SMART 198: Offline Uncorrectable Sector Count
The total count of uncorrectable errors when reading or writing a sector; a rise indicates defects of the disk surface and/or problems in the mechanical subsystem [1]. In the Google field study, drives were on average 39 times more likely to fail in the 60 days after their first offline uncorrectable error than similar drives without one ([3], as summarized in [1]). SMART 197 and 198 are well correlated in Backblaze data, but not every manufacturer reports both [2].

## SMART 187: Reported Uncorrectable Errors
The count of errors that could not be recovered using hardware ECC [1]. It is one of the five attributes Backblaze uses to flag drives [2]. Because it is cumulative, the rate matters: a jump from zero to 20 in one day is more alarming than 60 errors spread over five years [2].

## SMART 184: End-to-End error / IOEDC
A count of parity errors in the data path to the media via the drive cache RAM, part of HP SMART IV technology and of other vendors' IO error detection schemes [1]. Not every drive model reports it; in Backblaze data it is present only for some models, so its absence is not evidence of health.

## Interpreting media signals together
Backblaze found that 76.7% of failed drives had at least one of SMART 5, 187, 188, 197 or 198 above zero, versus 4.2% of operational drives; so 23.3% of failed drives gave no warning in these attributes [2]. Growth within a short window is more informative than a static non-zero count [2].
