A Python code to calculate in-situ Vp/Vs ratio using pairs of P and S differential travel times from a tight cluster of events. The differential times can be provided from FDTCC (cross-correlation-based differential times, recommended) or hypoDD (catalog-based differential times) programs. A robust L1-L2 fitting algorithm from Lin and Shearer (2007) is used to calculate the ratio.

The FDTCC program can be found here:  https://github.com/MinLiu19/FDTCC  
The hypoDD program can be found here: https://github.com/fwaldhauser/HypoDD

**Input Format**  
InsVpVs works based on the dt.cc file derived from FDTCC (recommended) and the dt.ct file derived from hypoDD:  
```
dt.cc format:
# event_ID1     event_ID2    0
station     differential_time       CC    phase

dt.ct format:
# event_ID1     event_ID2
station   tt_1    tt_2    1.00    phase
```

If dt.ct file is used, differential times are calculated as ```tt_1 - tt_2```.
<div id="header" align="center">
  <img src='vpvs_ratio.jpg' width='600'>
</div>

