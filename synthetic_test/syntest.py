import random
import math
import matplotlib.pyplot as plt
import numpy as np


nevent = 50         # number of events in cluster
ec_size = 0.005     # how far events can be located from each other? max will be 2*ec_size in deg
maxnoise = 0.007    # max differential noise in sec
cdep = 6            # cluster depth range
dep_shift = 0.4     # max event depth deviation from cdep
add_outlier = True  # add outlier P differentials?
n_outlier = 0.01*nevent*5   # number of outliers. 5 is the number of stations
outsize = 0.1      # outlier size. maximum amount to be added to an observation to make an outlier
vp = 6.4
vpvs = 1.915       # True Vp/Vs ratio
out = open('syn.cc', 'w')
vs = vp / vpvs


def distance_3d(lat1, lon1, depth1, lat2, lon2, depth2):
    R = 6371  # Earth radius in km

    # convert degrees to radians
    φ1, λ1 = math.radians(lat1), math.radians(lon1)
    φ2, λ2 = math.radians(lat2), math.radians(lon2)

    # convert depth to radius (depth is positive downward)
    r1 = R - depth1
    r2 = R - depth2

    # ECEF coordinates
    x1 = r1 * math.cos(φ1) * math.cos(λ1)
    y1 = r1 * math.cos(φ1) * math.sin(λ1)
    z1 = r1 * math.sin(φ1)

    x2 = r2 * math.cos(φ2) * math.cos(λ2)
    y2 = r2 * math.cos(φ2) * math.sin(λ2)
    z2 = r2 * math.sin(φ2)

    # straight-line distance
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


stlist = []
with open('station.dat') as stas:
    for line in stas:
        lon, lat = map(float, line.split()[:2])
        sta = line.split()[3]
        stlist.append([lon, lat, sta])

print(stlist)


center = [-129.1, 47.95]
events = []
n = 1
while n <= nevent:
    int_lat = random.uniform(-ec_size, ec_size)
    int_lon = random.uniform(-ec_size, ec_size)
    int_dep = random.uniform(-dep_shift, dep_shift)
    evlon, evlat, evdep = center[0]+int_lon, center[1]+int_lat, cdep+int_dep
    events.append([evlon, evlat, evdep])
    n+=1

print(events)

no = 0
for i, eve1 in enumerate(events):
    for j, eve2 in enumerate(events):
        if eve1 == eve2: continue
        out.write(f'#\t{i}\t{j}\t0\n')
        pp = []
        ss = []
        stations = []
        for sta in stlist:
            stlon, stlat, stnm = sta
            d1 = distance_3d(eve1[1], eve1[0], eve1[2], stlat, stlon, 0)
            d2 = distance_3d(eve2[1], eve2[0], eve2[2], stlat, stlon, 0)
            tp1, tp2 = d1 / vp, d2 / vp
            ts1, ts2 = d1 / vs, d2 / vs
            pdiff = tp1 - tp2
            sdiff = ts1 - ts2

            # outlier?
            if add_outlier:
                n = np.random.uniform(0, 1)
                if n > 0.5 and no <= n_outlier:
                    noise = np.random.uniform(-outsize, outsize)
                    pdiff += noise
                    no+=1

            pp.append(pdiff)
            ss.append(sdiff) 
            stations.append(stnm)           
        
        # add noise
        noise_P = np.random.normal(0, maxnoise, size=len(pp))
        noise_S = np.random.normal(0, maxnoise*vpvs, size=len(ss))
        tP_noisy = pp + noise_P
        tS_noisy = ss + noise_S

        for p, s, st in zip(tP_noisy, tS_noisy, stations):
            out.write(f'{st}\t{round(p, 3)}\t1.00\tP\n')
            out.write(f'{st}\t{round(s, 3)}\t1.00\tS\n')

out.close()




st_lon = [s[0] for s in stlist]
st_lat = [s[1] for s in stlist]
st_dep = [0 for _ in stlist]   # stations are at surface (depth 0)

ev_lon = [e[0] for e in events]
ev_lat = [e[1] for e in events]
ev_dep = [e[2] for e in events]   # depth in km

# ---- Plot ----

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Stations
ax.scatter(st_lon, st_lat, st_dep, marker='^', s=80, label='Stations')

# Events
ax.scatter(ev_lon, ev_lat, ev_dep, marker='o', alpha=0.7, label='Events')

# Labels
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Depth (km)")

# Invert depth axis (optional, seismology style)
ax.invert_zaxis()

ax.legend()
plt.tight_layout()
plt.savefig('syntest_config.jpg', dpi=400)
plt.show()