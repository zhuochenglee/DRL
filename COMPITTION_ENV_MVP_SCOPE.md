# Competition Env MVP Scope

## Included
- Annual scheduling calendar: 120-day extraction, 200-day injection, 45-day balance
- Daily data-driven baseline pressure and rate profiles for injection and extraction
- Low-dimensional dispatch action: operating intensity and renewable usage ratio
- Renewable availability profile derived from Attachment 2 seasonal wind and solar information
- Dispatch-oriented economics: gas revenue during extraction and grid electricity cost
- Renewable-share accounting with a 20% target
- Simple storage inventory proxy to connect injection and extraction scheduling
- Rule-based smoke-test policy and optional SAC training entry point

## Excluded In This MVP
- Detailed station and process design
- H2S / CO2 processing and gas-quality engineering
- Produced-water treatment and liquid handling
- Detailed multi-well routing and full pipeline topology reconstruction
- Full engineering cost model beyond dispatch revenue and electricity cost
- Exact attachment-table replay for every well; this MVP uses trend-consistent baselines

## Main Simplifications
- Pressure and flow are represented by baseline envelopes plus dispatch adjustments, not full fluid mechanics
- Renewable resources are converted into a daily availability factor from seasonal characteristics rather than a measured hourly time series
- Storage behavior is represented by a normalized inventory state instead of a full reservoir model
- Injection source switching between L plant and M pipeline is not yet modeled as an action

## Natural Next Extensions
- Add source-selection action for injection scheduling
- Replace trend baselines with full parsed daily tables from Attachment 1
- Split environment/data/evaluation into separate modules once the MVP stabilizes
