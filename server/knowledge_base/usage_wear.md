# Usage and wear attributes (SMART 9, 12, 193, 241, 242)

Sources:
[1] Wikipedia, "Self-Monitoring, Analysis and Reporting Technology", attribute table. https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology
[2] Backblaze Drive Stats dataset card (Hugging Face). https://huggingface.co/datasets/backblaze/Drive_Stats

## SMART 9: Power-On Hours
Counts time in the powered-on state; depending on the manufacturer the unit may be hours, minutes or seconds [1]. A common reference lifetime is five years of continuous operation, about 43,800 hours [1]. Age alone is a weak predictor, but it shapes how other signals should be read.

## SMART 12: Power Cycle Count
The count of full power on/off cycles [1]. In a data center drives are rarely power-cycled, so jumps usually reflect maintenance events or chassis problems.

## SMART 193: Load/Unload Cycle Count
Counts load/unload cycles of the heads into the landing zone [1]. Ratings vary by product line: for example 600,000 cycles for WD VelociRaptor and 300,000 for WD Green drives, but only 50,000 for one desktop model [1]. Some drives report this under ID 225 instead [1].

## SMART 241 / 242: Total LBAs Written / Read
Total count of logical blocks written or read [1]. Units differ between models (some report 1 GiB units), and some tools show negative raw values because the field is 48 bits [1]. These counters describe workload; compare a drive with others of the same model, not across models.

## Normalizing usage across drive models
SMART values are reported as each drive model defines them, and the set of reported attributes changes between quarters [2]. This system therefore normalizes features per drive model before comparing drives.
