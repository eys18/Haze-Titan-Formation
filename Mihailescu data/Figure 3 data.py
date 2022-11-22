"""
Spatial variations of low mass negative ions in Titan's upper atmosphere
Data for Figure 3 (density distribution of the two species)
Teia Mihailescu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import random
import math

#### DATA ####

#### SPECIES 1 ####

TA_altitude=np.array([1334,1249,1195,1195,1174,1185,1230])  #in km
TA_latitude=np.array([-61,-69,-79,-79,-89,-77,-65]) #in degrees
TA_loctime=np.array([72,77,84,84,90,96,101]) #in degrees
TA_density=np.array([11.9,14.6,12.0,7.4,6.6,6.6,1.2]) #in cm^-3
TA_e_density=np.array([1880., 1635., 2049.,  772.,  892., 1328.,  841.]) #in cm^-3
TA_e_temp=np.array([0.0589, 0.029 , 0.0016, 0.0097, 0.0332, 0.0573, 0.0575]) #in eV (I think?)
TA = [TA_altitude, TA_latitude, TA_loctime, TA_density, TA_e_density, TA_e_temp]

T16_altitude=np.array([1278,1278,1152,1055,988,954,953,986,986,1052])
T16_latitude=np.array([63,63,65,67,69,71,73,76,76,79])
T16_loctime=np.array([170,170,158,146,134,123,114,106,106,100])
T16_density=np.array([1.2,7.2,9.4,7.1,4.4,3.5,4.8,7.3,11.2,2.0])
T16_e_density=np.array([ 807.,  786.,  869.,  704.,  543.,  430.,  771.,  857., 1077., 1094.])
T16_e_temp=np.array([0.0494, 0.0389, 0.0378, 0.0374, 0.0396, 0.0353, 0.0395, 0.033, 0.0444, 0.058  ])
T16 = [T16_altitude, T16_latitude, T16_loctime, T16_density, T16_e_density, T16_e_temp]

T17_altitude=np.array([1249,1140,1061,1014,999,999,1018,1070])
T17_latitude=np.array([58,55,51,47,44,44,40,37])
T17_loctime=np.array([-36,-28,-20,-13,-6,-6,0,5])
T17_density=np.array([10.9,18.4,17.6,18.4,18.3,19.7,20.8,8.1])
T17_e_density = np.array([2215., 2874., 3062., 3041., 2878., 3118., 2762.,   np.nan])
T17_e_temp = np.array([0.0627, 0.0509, 0.0524, 0.0547, 0.0473, 0.0541, 0.0556,    np.nan])
T17 = [T17_altitude, T17_latitude, T17_loctime, T17_density, T17_e_density, T17_e_temp]

T18_altitude=np.array([1341,1205,1097,1019,973,959,980,980,1033])
T18_latitude=np.array([78,80,82,84,87,89,86,86,82])
T18_loctime=np.array([133,118,106,98,92,87,84,84,81])
T18_density=np.array([1.0,2.1,2.2,2.9,2.9,3.2,4.4,10.3,3.9])
T18_e_density=np.array([ 902.,  705.,  786.,   np.nan,   np.nan,  868., 1058., 1692., 1712.])
T18_e_temp=np.array([0.0557, 0.0337, 0.0427,    np.nan,    np.nan, 0.0386, 0.0357, 0.0429, 0.0636])
T18 = [T18_altitude, T18_latitude, T18_loctime, T18_density, T18_e_density, T18_e_temp]

T19_altitude=np.array([1263,1263,1147,1059,1003,980,990,1033,1033])
T19_latitude=np.array([88,88,88,86,82,79,74,69,69])
T19_loctime=np.array([91,91,85,81,78,75,73,71,71])
T19_density=np.array([2.7,9.1,8.1,6.7,7.5,8.9,9.9,12.7,6.6])
T19_e_density = np.array([1843., 1767., 1350., 1498., np.nan, 1480., 1905., 1936., 1922.])
T19_e_temp=np.array([0.0765, 0.0548, 0.0377, 0.059 , np.nan, 0.0495, 0.0585, 0.0461, 0.0562])
T19 = [T19_altitude, T19_latitude, T19_loctime, T19_density, T19_e_density, T19_e_temp]

T21_altitude=np.array([1208,1110,1042,1005,1002,1002,1032,1094])
T21_latitude=np.array([37,34,31,27,22,22,15,8])
T21_loctime=np.array([139,135,132,128,125,125,121,118])
T21_density=np.array([1.0,1.2,1.4,1.3,1.0,0.8,1.3,1.3])
T21_e_density=np.array([378., 330., 259., 270., 320., 318., 363.,  np.nan])
T21_e_temp=np.array([0.0381, 0.0397, 0.0305, 0.031 , 0.0285, 0.026 , 0.023 , np.nan])
T21 = [T21_altitude, T21_latitude, T21_loctime, T21_density, T21_e_density, T21_e_temp]

T23_altitude=np.array([1209,1209,1111,1042,1006,1002,1032,1032,1094])
T23_latitude=np.array([57,57,51,46,39,32,25,25,17])
T23_loctime=np.array([48,48,48,48,48,48,48,48,48])
T23_density=np.array([21.3,19.8,17.3,14.1,12.6,16.3,17.0,17.7,3.8])
T23_e_density=np.array([3144., np.nan, 2206., 2076., 2033., 1889., 2106., 2685., 1844.])
T23_e_temp=np.array([1.7029e-01, np.nan, 3.3356e-02, 9.9562e-02, 6.2987e-02, 5.3027e-02, 3.1700e-02, 2.5998e-02, 8.6512e-02])
T23 = [T23_altitude, T23_latitude, T23_loctime, T23_density, T23_e_density, T23_e_temp]

T25_altitude=np.array([1215,1111,1040,1004,1005,1005,1042,1114])
T25_latitude=np.array([1,4,10,15,21,21,27,33])
T25_loctime=np.array([-177,-178,-179,179,179,179,178,177])
T25_density=np.array([13.6,13.1,11.1,11.4,10.7,14.0,8.3,1.0])
T25_e_density=np.array([516., 400., 301., 280., 343., 631., 676., 597.])
T25_e_temp=np.array([0.0451, 0.0509, 0.0478, 0.0561, 0.0421, 0.061 , 0.0485, 0.0503])
T25 = [T25_altitude, T25_latitude, T25_loctime, T25_density, T25_e_density, T25_e_temp]

T26_altitude=np.array([1605,1427,1275,1275,1151,1059,1002,980,995,1047,1133,1133])
T26_latitude=np.array([-5,0,4,4,10,15,21,27,34,39,45,45])
T26_loctime=np.array([-165,-166,-166,-166,-166,-166,-166,-166,-167,-167,-167,-167])
T26_density=np.array([2.1,3.4,9.8,11.0,11.1,10.5,12.6,13.1,15.4,15.1,6.6,5.8])
T26_e_density=np.array([287., 502., 452., 424., 240., 190., 228., 280., 427., 581.,  np.nan, 295.])
T26_e_temp=np.array([0.075, 0.05  , 0.0386, 0.0454, 0.0369, 0.0351, 0.0409, 0.0399, 0.0373, 0.0392, np.nan, 0.0477])
T26 = [T26_altitude, T26_latitude, T26_loctime, T26_density, T26_e_density, T26_e_temp]

T27_altitude=np.array([1231,1125,1052,1015,1015,1048,1048])
T27_latitude=np.array([15,20,26,32,32,44,44])
T27_loctime=np.array([-171,-172,-172,-173,-173,-174,-174])
T27_density=np.array([3.5,6.4,8.6,13.4,15.7,16.0,9.3])
T27_e_density=np.array([ 31., 266., 280., 302., 335., 455., 539.])
T27_e_temp=np.array([np.nan, 4.6579e-02, 5.0000e-02, 6.8329e-02, 5.7631e-02, 3.6864e-02, 2.8761e-02])
T27 = [T27_altitude, T27_latitude, T27_loctime, T27_density, T27_e_density, T27_e_temp]

T28_altitude=np.array([1284,1160,1069,1012,991,991,1006,1057])
T28_latitude=np.array([19,24,30,36,42,42,48,53])
T28_loctime=np.array([-177,-178,-179,179,178,178,176,174])
T28_density=np.array([9.9,18.0,12.6,11.9,11.9,9.0,4.9,3.1])
T28_e_density=np.array([621., 520., 215., 205., 238., 382., 812., 887.])
T28_e_temp=np.array([np.nan, 0.0349, 0.0129, 0.0305, 0.0162, 0.0353, 0.0427, 0.0322])
T28 = [T28_altitude, T28_latitude, T28_loctime, T28_density, T28_e_density, T28_e_temp]

T29_altitude=np.array([1270,1147,1057,1057,1001,980,997,1050])
T29_latitude=np.array([26,32,37,37,43,49,55,61])
T29_loctime=np.array([175,174,172,172,170,168,164,160])
T29_density=np.array([17.2,27.4,17.8,14.6,7.5,6.0,8.0,6.6])
T29_e_density=np.array([1146.,  849.,  541.,  450.,  499.,  787., 1064.,  959.])
T29_e_temp=np.array([0.0489, 0.0459, 0.0364, 0.0522, 0.053 , 0.0416, 0.0546, 0.0331])
T29 = [T29_altitude, T29_latitude, T29_loctime, T29_density, T29_e_density, T29_e_temp]

T30_altitude=np.array([1036,1036,959,959])
T30_latitude=np.array([44,44,56,56])
T30_loctime=np.array([162,162,155,155])
T30_density=np.array([5.5,4.5,4.4,3.6])
T30_e_density=np.array([287., 358., 546., 607.])
T30_e_temp=np.array([0.06, 0.0542, 0.0393, 0.0458])
T30 = [T30_altitude, T30_latitude, T30_loctime, T30_density, T30_e_density, T30_e_temp]

T32_altitude=np.array([1286,1286,1156,1058,994,966,974,1019,1019])
T32_latitude=np.array([45,45,50,56,63,69,76,82,82])
T32_loctime=np.array([151,151,147,142,135,127,116,103,103])
T32_density=np.array([5.6,8.3,5.9,3.9,3.2,4.2,7.3,11.9,11.9])
T32_e_density=np.array([ 747.,  710.,  421.,  343.,  319.,  340.,  502.,  797., 1251.])
T32_e_temp=np.array([0.0304, 0.0306, 0.0297, 0.0383, 0.0418, 0.0355, 0.0349, 0.0366, 0.0539])
T32 = [T32_altitude, T32_latitude, T32_loctime, T32_density, T32_e_density, T32_e_temp]

T36_altitude=np.array([1382,1382,1235,1117,1033,985,973,1000,1000,1062,1160])
T36_latitude=np.array([-86,-86,-88,-82,-77,-71,-66,-60,-60,-55,-49])
T36_loctime=np.array([110,110,87,55,29,13,4,-1,-1,-5,-8])
T36_density=np.array([3.0,11.7,20.3,21.0,10.2,12.2,15.3,27.9,21.9,10.5,2.7])
T36_e_density=np.array([ 984., 1310., 1626., 1672., 1707., 1554., 1847., 2200., 2013., 1518.,  855.])
T36_e_temp=np.array([1.54  , 0.1881, 0.0508, 0.0362, 0.0363, 0.0427, 0.0445, 0.0484, 0.0536, 0.0627, 0.0171]) 
T36 = [T36_altitude, T36_latitude, T36_loctime, T36_density, T36_e_density, T36_e_temp]

T39_altitude=np.array([1261,1136,1136,1044,988,969,988,1044,1136])
T39_latitude=np.array([-79,-74,-74,-69,-63,-57,-51,-45,-38])
T39_loctime=np.array([-57,-52,-52,-48,-45,-43,-41,-40,-39])
T39_density=np.array([14.0,26.1,17.0,13.3,15.2,18.7,21.4,22.9,5.1])
T39_e_density=np.array([2280., 2428., 1733., 1517., 1545., 1836., 2071., 2344., 1404.])
T39_e_temp=np.array([0.0547, 0.044 , 0.0664, 0.0528, 0.0532, 0.0484, 0.0481, 0.054, 0.0784])
T39 = [T39_altitude, T39_latitude, T39_loctime, T39_density, T39_e_density, T39_e_temp]

T40_altitude=np.array([1360,1225,1121,1051,1017,1017,1020,1060,1137])
T40_latitude=np.array([-52,-45,-39,-32,-26,-26,-20,-14,-9])
T40_loctime=np.array([53,48,43,38,34,34,30,26,22])
T40_density=np.array([11.0,28.6,25.6,21.8,22.2,25.7,32.5,23.7,3.3])
T40_e_density=np.array([1652., 2549., 2488., 2400., 2400., 2599., 2784., 2285., 1230.])
T40_e_temp=np.array([0.07  , 0.0513, 0.051 , 0.0476, 0.0451, 0.0427, 0.0542, 0.053 , 0.0717])
T40 = [T40_altitude, T40_latitude, T40_loctime, T40_density, T40_e_density, T40_e_temp]

T41_altitude=np.array([1318,1187,1089,1026,1000,1000,1011,1060])
T41_latitude=np.array([-51,-47,-41,-36,-31,-31,-25,-20])
T41_loctime=np.array([16,11,6,2,0,0,-3,-6])
T41_density=np.array([9.4,29.0,32.8,29.6,30.2,30.4,24.6,4.7])
T41_e_density=np.array([1589., 2826., 3090., 2926., 2900., 3029., 2444., 1453.])
T41_e_temp=np.array([0.0668, 0.0569, 0.0444, 0.0411, 0.057 , 0.0483, 0.057 , 0.0671])
T41 = [T41_altitude, T41_latitude, T41_loctime, T41_density, T41_e_density, T41_e_temp]

T42_altitude=np.array([1425,1274,1153,1153,1065,1014,999,1022,1083,1083])
T42_latitude=np.array([-46,-41,-36,-36,-31,-26,-20,-15,-9,-9])
T42_loctime=np.array([15,11,7,7,4,0,-2,-5,-7,-7])
T42_density=np.array([3.3,11.0,24.4,26.7,25.7,26.1,27.5,21.7,12.6,3.9])
T42_e_density=np.array([1319., 2098., 2556., 2827., 2815., 2752., 2643., 2508., 2075., 788.])
T42_e_temp=np.array([0.0736, 0.0669, 0.04  , 0.04  , 0.04  , 0.04  , 0.03  , 0.0315, 0.04  , 0.0594])
T42 = [T42_altitude, T42_latitude, T42_loctime, T42_density, T42_e_density, T42_e_temp]

T43_altitude=np.array([1298,1172,1079,1022,1001,1018,1018])
T43_latitude=np.array([-14,-7,-1,5,11,17,17])
T43_loctime=np.array([46,43,40,37,34,31,31])
T43_density=np.array([19.7,32.7,32.4,34.1,28.8,34.1,19.1])
T43_e_density=np.array([2064., 2616., 2544., 2813., 2368., 2167., 1954.])
T43_e_temp=np.array([0.0348, 0.0374, 0.0295, 0.0269, 0.0298, 0.0316, 0.0323])
T43 = [T43_altitude, T43_latitude, T43_loctime, T43_density, T43_e_density, T43_e_temp]

T46_altitude=np.array([1213,1143,1109,1147,1147])
T46_latitude=np.array([-20,-15,-10,0,0])
T46_loctime=np.array([172,175,178,-175,-175])
T46_density=np.array([0.4,6.8,8.5,2.8,3.6])
T46_e_density=np.array([889., 793., 746.,   np.nan, 326.])
T46_e_temp=np.array([0.0437, 0.0429, 0.0157,    np.nan,    np.nan])
T46 = [T46_altitude, T46_latitude, T46_loctime, T46_density, T46_e_density, T46_e_temp]

T48_altitude=np.array([1347,1226,1108,1023,973,973,961,986,1047])
T48_latitude=np.array([-19,-15,-10,-5,0,0,6,12,18])
T48_loctime=np.array([-11,-14,-16,-19,-22,-22,-25,-27,-30])
T48_density=np.array([4.8,23.4,38.0,37.9,33.9,30.2,32.0,29.0,8.0])
T48_e_density=np.array([1438., 2399., 2558., 2541., 2453., 2367., 2448., 2237., 1991.])
T48_e_temp=np.array([0.0774, 0.0608, 0.0447, 0.0419, 0.0408, 0.0386, 0.0408, 0.0488, 0.0731])
T48 = [T48_altitude, T48_latitude, T48_loctime, T48_density, T48_e_density, T48_e_temp]

T49_altitude=np.array([1251,1251,1129,1041,987,970,991,1048])
T49_latitude=np.array([-67,-67,-67,-66,-65,-64,-62,-54])
T49_loctime=np.array([-73,-73,-75,-78,-80,-82,-85,-86])
T49_density=np.array([6.2,17.0,9.0,7.8,6.1,5.3,5.3,6.6])
T49_e_density=np.array([1752., 2182., 1592., 1103., 1140., 1185., 1619., 2153.])
T49_e_temp=np.array([0.0555, 0.0305, 0.0285, 0.0306, 0.0273, 0.047 , 0.032 , 0.0413])
T49 = [T49_altitude, T49_latitude, T49_loctime, T49_density, T49_e_density, T49_e_temp]

T50_altitude=np.array([1250,1250,1127,1038,984,966,986,986,1043])
T50_latitude=np.array([-51,-51,-46,-39,-32,-25,-18,-18,-10])
T50_loctime=np.array([-140,-140,-140,-140,-140,-140,-141,-141,-141])
T50_density=np.array([1.7,3.3,12.9,15.9,13.4,12.2,11.4,8.5,5.7])
T50_e_density=np.array([ 729.,  890.,  952.,  695.,  640.,  581., 1023.,  990.,  736.])
T50_e_temp=np.array([0.0248, 0.0258, 0.0096, 0.0212, 0.0187, 0.0126, 0.0193, 0.0142, 0.0456])
T50 = [T50_altitude, T50_latitude, T50_loctime, T50_density, T50_e_density, T50_e_temp]

T51_altitude=np.array([1355,1211,1097,1016,971,964,993,1060])
T51_latitude=np.array([-65,-63,-60,-55,-43,-7,54,79])
T51_loctime=np.array([-77,-78,-80,-81,-83,-84,-86,-87])
T51_density=np.array([3.9,18.2,14.1,13.9,12.3,12.0,12.7,9.6])
T51_e_density=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
T51_e_temp=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
T51 = [T51_altitude, T51_latitude, T51_loctime, T51_density, T51_e_density, T51_e_temp]

T65_altitude=np.array([1414,1288,1189,1120,1081,1081,1075,1101])
T65_latitude=np.array([-55,-61,-68,-74,-79,-79,-85,-89])
T65_loctime=np.array([-118,-116,-113,-109,-105,-105,-100,-93])
T65_density=np.array([0.8,1.1,2.5,3.1,5.1,6.0,5.5,3.2])
T65_e_density=np.array([ 475.,  579.,  719.,  757., 1035., 1316., 1411., 1387.])
T65_e_temp=np.array([0.0506, 0.0436, 0.0354, 0.0348, 0.0372, 0.0406, 0.0419, 0.0599])
T65 = [T65_altitude, T65_latitude, T65_loctime, T65_density, T65_e_density, T65_e_temp]

T71_altitude=np.array([1372,1240,1135,1059,1015,1015,1004,1025,1078])
T71_latitude=np.array([-50,-63,-76,-87,-82,-82,-74,-68,-62])
T71_loctime=np.array([105,101,97,92,87,87,81,75,69])
T71_density=np.array([2.1,6.6,11.5,10.2,9.2,11.7,15.0,14.2,4.6])
T71_e_density=np.array([ 977., 1255., 1263., 1269., 1302., 1684., 2133., 2467., 1695.])
T71_e_temp=np.array([0.0597, 0.0415, 0.0296, 0.0302, 0.0287, 0.0356,    np.nan, 0.0436, 0.0366])
T71 = [T71_altitude, T71_latitude, T71_loctime, T71_density, T71_e_density, T71_e_temp]

T83_altitude=np.array([1241,1125,1037,980,954,961,961,1001])
T83_latitude=np.array([84,89,82,75,67,60,60,54])
T83_loctime=np.array([-93,-85,-77,-69,-61,-54,-54,-48])
T83_density=np.array([12.6,16.5,13.4,12.9,15.1,26.6,24.9,13.5])
T83_e_density=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
T83_e_temp=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
T83 = [T83_altitude, T83_latitude, T83_loctime, T83_density, T83_e_density, T83_e_temp]

flybys = [TA, T16, T17, T18, T19, T21, T23, T25, T26, T27, T28, T29, T30, T32, T36, T39, T40, T41, T42, T43, T46, T48, T49, T50, T51, T65, T71, T83]
names = ['TA', 'T16', 'T17', 'T18', 'T19', 'T21', 'T23', 'T25', 'T26', 'T27', 'T28', 'T29', 'T30', 'T32', 'T36', 'T39', 'T40', 'T41', 'T42', 'T43', 'T46', 'T48', 'T49', 'T50', 'T51', 'T65', 'T71', 'T83']

colors = ['#C1F09C', '#E535B9', '#A34D8C', '#ADB230', '#78F4F1', '#59C054', '#597DFD', '#D1039E', '#647EBA', '#A9A5A1', '#E9495B', '#DF9DD1', '#574364', '#265B6C', '#FD7BCA', '#C343BD', '#26E630', '#69A0A6', '#64C100', '#EA7CF9', '#C10965', '#1B882F', '#97969D', '#0894B0', '#6DD907', '#8E37CB', '#C62B8E', '#C2E136']

class altitude_layer:
    def __init__(self, altitude, latitude, local_time, density):
        self.altitude = altitude
        self.latitude = latitude
        self.local_time = local_time
        self.density = density
        
layer1 = altitude_layer([], [], [], [])
layer2 = altitude_layer([], [], [], [])
layer3 = altitude_layer([], [], [], [])
layer4 = altitude_layer([], [], [], [])
layer5 = altitude_layer([], [], [], [])

for i in range(0, len(flybys)):
    for j in range(0, len(flybys[i][0])):
        if(flybys[i][0][j]>950 and flybys[i][0][j]<=1050):
            layer1.altitude = np.append(layer1.altitude, int(flybys[i][0][j]))
            layer1.latitude = np.append(layer1.latitude, int(flybys[i][1][j]))
            layer1.local_time = np.append(layer1.local_time, int(flybys[i][2][j]))
            layer1.density = np.append(layer1.density, int(flybys[i][3][j]))
            
        if(flybys[i][0][j]>1050 and flybys[i][0][j]<=1150):
            layer2.altitude = np.append(layer2.altitude, int(flybys[i][0][j]))
            layer2.latitude = np.append(layer2.latitude, int(flybys[i][1][j]))
            layer2.local_time = np.append(layer2.local_time, int(flybys[i][2][j]))
            layer2.density = np.append(layer2.density, int(flybys[i][3][j]))
            
        if(flybys[i][0][j]>1150 and flybys[i][0][j]<=1250):
            layer3.altitude = np.append(layer3.altitude, int(flybys[i][0][j]))
            layer3.latitude = np.append(layer3.latitude, int(flybys[i][1][j]))
            layer3.local_time = np.append(layer3.local_time, int(flybys[i][2][j]))
            layer3.density = np.append(layer3.density, int(flybys[i][3][j]))
            
        if(flybys[i][0][j]>1250 and flybys[i][0][j]<=1350):
            layer4.altitude = np.append(layer4.altitude, int(flybys[i][0][j]))
            layer4.latitude = np.append(layer4.latitude, int(flybys[i][1][j]))
            layer4.local_time = np.append(layer4.local_time, int(flybys[i][2][j]))
            layer4.density = np.append(layer4.density, int(flybys[i][3][j]))
        
        if(flybys[i][0][j]>1350):
            layer5.altitude = np.append(layer5.altitude, int(flybys[i][0][j]))
            layer5.latitude = np.append(layer5.latitude, int(flybys[i][1][j]))
            layer5.local_time = np.append(layer5.local_time, int(flybys[i][2][j]))
            layer5.density = np.append(layer5.density, int(flybys[i][3][j]))


plt.subplots(4, 2, figsize=(24,24))

plt.subplot(427)
plt.scatter(layer1.local_time, layer1.latitude,linewidth=3.0, c=layer1.density*0.25, vmin=0, vmax=10, s=100, cmap='plasma')
plt.title('Altitude range 950-1050 km', fontsize=19.0)
plt.ylim(-90,90)
plt.ylabel('Latitude (\xb0N)', fontsize=19.0)
plt.xlim(-180,180)
plt.xlabel('Titan Local Time', fontsize=19.0)
plt.xticks([-180, -120, -60, 0, 60, 120, 180], ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=16.0)
plt.yticks([-90, -60, -30, 0, 30, 60, 90], ['-90', '-60', '-30', '0', '30', '60', '90'], fontsize=16.0)
cb = plt.colorbar()
cb.set_label(label='Normalised density ($cm^{-3} \cdot \u03B5$)', fontsize=19.0)
cb.ax.tick_params(labelsize=16.0)
plt.text(-210, 105, 'd)', fontsize=25, fontweight='bold', va='top', ha='right')
plt.text(-168, 70, '$CN^-$ / $C_2H^-$', bbox=dict(boxstyle="round", facecolor='none'), fontsize=20.0)

plt.plot([-90, -90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([90, 90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([-138, -138], [-90, 90], c='lightgray', linestyle='dotted')
plt.plot([138, 138], [-90, 90], c='lightgray', linestyle='dotted')

plt.subplot(425)
plt.scatter(layer2.local_time, layer2.latitude,linewidth=3.0, c=layer2.density*0.25, vmin=0, vmax=10, s=100, cmap='plasma')
plt.title('Altitude range 1050-1150 km', fontsize=19.0)
plt.ylim(-90,90)
plt.ylabel('Latitude (\xb0N)', fontsize=19.0)
plt.xlim(-180,180)
#plt.xlabel('Titan Local Time', fontsize=19.0)
plt.xticks([-180, -120, -60, 0, 60, 120, 180], ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=16.0)
plt.yticks([-90, -60, -30, 0, 30, 60, 90], ['-90', '-60', '-30', '0', '30', '60', '90'], fontsize=16.0)
cb = plt.colorbar()
cb.set_label(label='Normalised density ($cm^{-3} \cdot \u03B5$)', fontsize=19.0)
cb.ax.tick_params(labelsize=16.0)
plt.text(-210, 105, 'c)', fontsize=25, fontweight='bold', va='top', ha='right')
plt.text(-168, 70, '$CN^-$ / $C_2H^-$', bbox=dict(boxstyle="round", facecolor='none'), fontsize=20.0)

plt.plot([-90, -90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([90, 90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([-138, -138], [-90, 90], c='lightgray', linestyle='dotted')
plt.plot([138, 138], [-90, 90], c='lightgray', linestyle='dotted')

plt.subplot(423)
plt.scatter(layer3.local_time, layer3.latitude,linewidth=3.0, c=layer3.density*0.25, vmin=0, vmax=10, s=100, cmap='plasma')
plt.title('Altitude range 1150-1250 km', fontsize=19.0)
plt.ylim(-90,90)
plt.ylabel('Latitude (\xb0N)', fontsize=19.0)
plt.xlim(-180,180)
#plt.xlabel('Titan Local Time', fontsize=19.0)
plt.xticks([-180, -120, -60, 0, 60, 120, 180], ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=16.0)
plt.yticks([-90, -60, -30, 0, 30, 60, 90], ['-90', '-60', '-30', '0', '30', '60', '90'], fontsize=16.0)
cb = plt.colorbar()
cb.set_label(label='Normalised density ($cm^{-3} \cdot \u03B5$)', fontsize=19.0)
cb.ax.tick_params(labelsize=16.0)
plt.text(-210, 105, 'b)', fontsize=25, fontweight='bold', va='top', ha='right')
plt.text(-168, 70, '$CN^-$ / $C_2H^-$', bbox=dict(boxstyle="round", facecolor='none'), fontsize=20.0)

plt.plot([-90, -90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([90, 90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([-138, -138], [-90, 90], c='lightgray', linestyle='dotted')
plt.plot([138, 138], [-90, 90], c='lightgray', linestyle='dotted')

plt.subplot(421)
plt.scatter(layer4.local_time, layer4.latitude,linewidth=3.0, c=layer4.density*0.25, vmin=0, vmax=10, s=100, cmap='plasma')
plt.title('Altitude range 1250-1350 km', fontsize=19.0)
plt.ylim(-90,90)
plt.ylabel('Latitude (\xb0N)', fontsize=19.0)
plt.xlim(-180,180)
#plt.xlabel('Titan Local Time', fontsize=19.0)
plt.xticks([-180, -120, -60, 0, 60, 120, 180], ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=16.0)
plt.yticks([-90, -60, -30, 0, 30, 60, 90], ['-90', '-60', '-30', '0', '30', '60', '90'], fontsize=16.0)
cb = plt.colorbar()
cb.set_label(label='Normalised density ($cm^{-3} \cdot \u03B5$)', fontsize=19.0)
cb.ax.tick_params(labelsize=16.0)
plt.text(-168, 70, '$CN^-$ / $C_2H^-$', bbox=dict(boxstyle="round", facecolor='none'), fontsize=20.0)
plt.text(-210, 105, 'a)', fontsize=25, fontweight='bold', va='top', ha='right')

plt.plot([-90, -90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([90, 90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([-138, -138], [-90, 90], c='lightgray', linestyle='dotted')
plt.plot([138, 138], [-90, 90], c='lightgray', linestyle='dotted')





### SPECIES 2 ####


TA_altitude=np.array([1334,1249,1195,1195,1174,1185,1230])
TA_latitude=np.array([-61,-69,-79,-79,-89,-77,-65])
TA_loctime=np.array([72,77,84,84,90,96,101])
TA_density=np.array([9.5,12.1,8.8,4.4,3.6,3.9,1.3])
TA_e_density=np.array([1881., 1635., 2050.,  731.,  882., 1329.,  812.])
TA_e_temp=np.array([0.0589, 0.029 , 0.0016, 0.0075, 0.0332, 0.0573, 0.0556])
TA = [TA_altitude, TA_latitude, TA_loctime, TA_density, TA_e_density, TA_e_temp]

T16_altitude=np.array([1278,1278,1152,1055,988,954,953,986,986,1052])
T16_latitude=np.array([63,63,65,67,69,71,73,76,76,79])
T16_loctime=np.array([170,170,158,146,134,123,114,106,106,100])
T16_density=np.array([0.6,7.1,10.7,8.3,6.1,4.9,6.7,8.1,6.8,0.9])
T16_e_density=np.array([ 776.,  784.,  870.,  681.,  553.,  428.,  772.,  858., 1078., 1095.])
T16_e_temp=np.array([0.0501, 0.0381, 0.0379, 0.0375, 0.0395, 0.0339, 0.0395, 0.033 , 0.0444, 0.058 ])
T16 = [T16_altitude, T16_latitude, T16_loctime, T16_density, T16_e_density, T16_e_temp]

T17_altitude=np.array([1249,1140,1061,1014,999,999,1018,1070])
T17_latitude=np.array([58,55,51,47,44,44,40,37])
T17_loctime=np.array([-36,-28,-20,-13,-6,-6,0,5])
T17_density=np.array([11.1,18.2,15.5,13.0,13.7,13.8,13.4,7.0])
T17_e_density=np.array([2215., 2874., 3062., 3041., 2878., 3141., 2762.,   np.nan])
T17_e_temp=np.array([0.0627, 0.0509, 0.0524, 0.0547, 0.0473, 0.0549, 0.0556,    np.nan])
T17 = [T17_altitude, T17_latitude, T17_loctime, T17_density, T17_e_density, T17_e_temp]

T18_altitude=np.array([1341,1205,1097,1019,973,959,980,980,1033])
T18_latitude=np.array([78,80,82,84,87,89,86,86,82])
T18_loctime=np.array([133,118,106,98,92,87,84,84,81])
T18_density=np.array([1.1,2.0,2.3,2.8,3.0,3.2,4.5,10.7,4.1])
T18_e_density=np.array([ 902.,  717.,  768.,  738.,   np.nan,  864., 1082., 1692., 1712.])
T18_e_temp=np.array([0.0537, 0.0337, 0.0415, 0.0376,    np.nan, 0.0384, 0.0362, 0.0429, 0.0636])
T18 = [T18_altitude, T18_latitude, T18_loctime, T18_density, T18_e_density, T18_e_temp]

T19_altitude=np.array([1263,1263,1147,1059,1003,980,990,1033,1033])
T19_latitude=np.array([88,88,88,86,82,79,74,69,69])
T19_loctime=np.array([91,91,85,81,78,75,73,71,71])
T19_density=np.array([2.3,8.1,6.9,4.9,5.3,5.0,6.6,9.1,5.6])
T19_e_density=np.array([1786., 1767., 1372., 1499.,   np.nan, 1480., 1905., 1936., 1922.])
T19_e_temp=np.array([0.0767, 0.0548, 0.0384, 0.0594,    np.nan, 0.0495, 0.0585, 0.0461, 0.0562])
T19 = [T19_altitude, T19_latitude, T19_loctime, T19_density, T19_e_density, T19_e_temp]

T21_altitude=np.array([1208,1110,1042,1005,1002,1002,1032,1094])
T21_latitude=np.array([37,34,31,27,22,22,15,8])
T21_loctime=np.array([139,135,132,128,125,125,121,118])
T21_density=np.array([0.7,1.1,1.1,1.0,0.6,0.5,0.8,0.6])
T21_e_density=np.array([380., 327., 259., 275., 321., 319., 363.,  np.nan])
T21_e_temp=np.array([0.038 , 0.0398, 0.0305, 0.0308, 0.0286, 0.026 , 0.023 ,    np.nan])
T21 = [T21_altitude, T21_latitude, T21_loctime, T21_density, T21_e_density, T21_e_temp]

T23_altitude=np.array([1209,1209,1111,1042,1006,1002,1032,1032,1094])
T23_latitude=np.array([57,57,51,46,39,32,25,25,17])
T23_loctime=np.array([48,48,48,48,48,48,48,48,48])
T23_density=np.array([19.7,18.0,14.8,11.7,11.2,13.4,15.0,17.1,2.7])
T23_e_density=np.array([3144.,   np.nan, 2206., 2076., 2033., 1889., 2094., 2685., 1821.])
T23_e_temp=np.array([1.7029e-01, np.nan, 3.3356e-02, 9.9562e-02, 6.2987e-02, 5.3027e-02, 3.2845e-02, 2.5998e-02, 8.8245e-02])
T23 = [T23_altitude, T23_latitude, T23_loctime, T23_density, T23_e_density, T23_e_temp]

T25_altitude=np.array([1215,1111,1040,1004,1005,1005,1042,1114])
T25_latitude=np.array([1,4,10,15,21,21,27,33])
T25_loctime=np.array([-177,-178,-179,179,179,179,178,177])
T25_density=np.array([7.9,8.7,8.2,8.2,7.9,8.4,4.2,0.3])
T25_e_density=np.array([554., 400., 306., 286., 342., 670., 676., 594.])
T25_e_temp=np.array([0.0451, 0.054 , 0.0485, 0.0559, 0.0423, 0.0613, 0.0485, 0.0497])
T25 = [T25_altitude, T25_latitude, T25_loctime, T25_density, T25_e_density, T25_e_temp]

T26_altitude=np.array([1605,1427,1275,1275,1151,1059,1002,980,995,1047,1133,1133])
T26_latitude=np.array([-5,0,4,4,10,15,21,27,34,39,45,45])
T26_loctime=np.array([-165,-166,-166,-166,-166,-166,-166,-166,-167,-167,-167,-167])
T26_density=np.array([0.2,1.1,4.8,6.8,8.6,8.4,10.4,10.4,9.4,8.2,4.6,2.4])
T26_e_density=np.array([244., 519., 458., 425., 240., 191., 227., 280., 422., 593.,  np.nan, 293.])
T26_e_temp=np.array([0.0799, 0.05  , 0.0387, 0.0463, 0.0369, 0.0354, 0.0411, 0.0399, 0.0377, 0.04  ,  np.nan, 0.0481])
T26 = [T26_altitude, T26_latitude, T26_loctime, T26_density, T26_e_density, T26_e_temp]

T27_altitude=np.array([1231,1125,1052,1015,1013,1048,1048])
T27_latitude=np.array([15,20,26,32,38,44,44])
T27_loctime=np.array([-171,-172,-172,-173,-174,-174,-174])
T27_density=np.array([6.2,8.8,10.5,11.1,11.2,10.2,5.6])
T27_e_density=np.array([ 71., 266., 280., 296., 342., 420., 491.])
T27_e_temp=np.array([np.nan, 4.6579e-02, 5.0746e-02, 6.9000e-02, 5.3881e-02, 4.0737e-02, 3.1803e-02])
T27 = [T27_altitude, T27_latitude, T27_loctime, T27_density, T27_e_density, T27_e_temp]

T28_altitude=np.array([1284,1160,1069,1012,991,991,1006,1057])
T28_latitude=np.array([19,24,30,36,42,42,48,53])
T28_loctime=np.array([-177,-178,-179,179,178,178,176,174])
T28_density=np.array([5.2,11.6,10.6,11.0,9.6,7.3,3.2,1.8])
T28_e_density=np.array([584., 484., 218., 206., 232., 390., 806., 887.])
T28_e_temp=np.array([   np.nan, 0.0376, 0.0135, 0.0294, 0.0207, 0.0342, 0.0431, 0.0322])
T28 = [T28_altitude, T28_latitude, T28_loctime, T28_density, T28_e_density, T28_e_temp]

T29_altitude=np.array([1270,1147,1057,1057,1001,980,997,1050])
T29_latitude=np.array([26,32,37,37,43,49,55,61])
T29_loctime=np.array([175,174,172,172,170,168,164,160])
T29_density=np.array([9.6,13.7,13.0,11.9,5.6,4.2,5.2,3.8])
T29_e_density=np.array([1182.,  846.,  544.,  444.,  499.,  821., 1069.,  959.])
T29_e_temp=np.array([0.0491, 0.0464, 0.0355, 0.0526, 0.053 , 0.0439, 0.0563, 0.0331])
T29 = [T29_altitude, T29_latitude, T29_loctime, T29_density, T29_e_density, T29_e_temp]

T30_altitude=np.array([1036,979,959,959])
T30_latitude=np.array([44,50,56,56])
T30_loctime=np.array([162,159,155,155])
T30_density=np.array([4.8,3.7,3.8,3.4])
T30_e_density=np.array([275., 400., 550., 612.])
T30_e_temp=np.array([0.06  , 0.05  , 0.04  , 0.0464])
T30 = [T30_altitude, T30_latitude, T30_loctime, T30_density, T30_e_density, T30_e_temp]

T32_altitude=np.array([1286,1286,1156,1058,994,966,974,1019,1019])
T32_latitude=np.array([45,45,50,56,63,69,76,82,82])
T32_loctime=np.array([151,151,147,142,135,127,116,103,103])
T32_density=np.array([4.2,7.5,5.9,4.4,3.5,4.3,7.0,8.4,7.5])
T32_e_density=np.array([ 406.,  672.,  421.,  338.,  317.,  342.,  502.,  797., 1251.])
T32_e_temp=np.array([0.0302, 0.0317, 0.0297, 0.0383, 0.0417, 0.035 , 0.0349, 0.0366, 0.0539])
T32 = [T32_altitude, T32_latitude, T32_loctime, T32_density, T32_e_density, T32_e_temp]

T36_altitude=np.array([1382,1382,1235,1117,1033,985,973,1000,1000,1062,1160])
T36_latitude=np.array([-86,-86,-88,-82,-77,-71,-66,-60,-60,-55,-49])
T36_loctime=np.array([110,110,87,55,29,13,4,-1,-1,-5,-8])
T36_density=np.array([2.1,7.1,10.8,12.5,8.4,8.7,9.2,16.3,13.4,4.9,1.1])
T36_e_density=np.array([ 974., 1328., 1626., 1677., 1714., 1554., 1831., 2200., 2033., 1495.,  878.])
T36_e_temp=np.array([2.1757, 0.1797, 0.0508, 0.0359, 0.0363, 0.0427, 0.0451, 0.0484, 0.0535, 0.063 , 0.0131])
T36 = [T36_altitude, T36_latitude, T36_loctime, T36_density, T36_e_density, T36_e_temp]

T39_altitude=np.array([1261,1136,1136,1044,988,969,988,1044,1136])
T39_latitude=np.array([-79,-74,-74,-69,-63,-57,-51,-45,-38])
T39_loctime=np.array([-57,-52,-52,-48,-45,-43,-41,-40,-39])
T39_density=np.array([13.6,24.9,13.4,11.0,11.7,16.1,21.0,24.0,5.7])
T39_e_density=np.array([2234., 2403., 1757., 1507., 1545., 1836., 2084., 2344., 1360.])
T39_e_temp=np.array([0.0559, 0.045 , 0.0668, 0.0525, 0.0532, 0.0484, 0.0484, 0.054 , 0.0813])
T39 = [T39_altitude, T39_latitude, T39_loctime, T39_density, T39_e_density, T39_e_temp]

T40_altitude=np.array([1360,1225,1121,1051,1017,1017,1020,1060,1137])
T40_latitude=np.array([-52,-45,-39,-32,-26,-26,-20,-14,-9])
T40_loctime=np.array([53,48,43,38,34,34,30,26,22])
T40_density=np.array([6.3,22.6,19.9,17.1,18.6,22.0,27.7,17.2,1.8])
T40_e_density=np.array([1652., 2549., 2488., 2400., 2400., 2616., 2784., 2285., 1230.])
T40_e_temp=np.array([0.07  , 0.0513, 0.051 , 0.0474, 0.0454, 0.0436, 0.0542, 0.053 , 0.0717])
T40 = [T40_altitude, T40_latitude, T40_loctime, T40_density, T40_e_density, T40_e_temp]

T41_altitude=np.array([1318,1187,1089,1026,1000,1000,1011,1060])
T41_latitude=np.array([-51,-47,-41,-36,-31,-31,-25,-20])
T41_loctime=np.array([16,11,6,2,0,0,-3,-6])
T41_density=np.array([8.6,29.6,39.2,31.6,35.0,39.9,27.0,4.5])
T41_e_density=np.array([1442., 2870., 3090., 2926., 2900., 3029., 2444., 1416.])
T41_e_temp=np.array([0.0678, 0.0564, 0.0444, 0.0411, 0.057 , 0.0483, 0.057 , 0.0685])
T41 = [T41_altitude, T41_latitude, T41_loctime, T41_density, T41_e_density, T41_e_temp]

T42_altitude=np.array([1425,1274,1153,1153,1065,1014,999,1022,1083,1083])
T42_latitude=np.array([-46,-41,-36,-36,-31,-26,-20,-15,-9,-9])
T42_loctime=np.array([15,11,7,7,4,0,-2,-5,-7,-7])
T42_density=np.array([1.4,6.4,15.8,16.8,15.5,14.6,16.7,15.7,9.7,2.1])
T42_e_density=np.array([1303., 2133., 2556., 2827., 2811., 2750., 2643., 2496., 2074., 782.])
T42_e_temp=np.array([0.0741, 0.066 , 0.04  , 0.04  , 0.04  , 0.04  , 0.03  , 0.0323, 0.04  , 0.0619])
T42 = [T42_altitude, T42_latitude, T42_loctime, T42_density, T42_e_density, T42_e_temp]

T43_altitude=np.array([1298,1172,1079,1022,1001,1018,1018])
T43_latitude=np.array([-14,-7,-1,5,11,17,17])
T43_loctime=np.array([46,43,40,37,34,31,31])
T43_density=np.array([15.0,34.1,33.7,39.9,29.9,32.0,14.4])
T43_e_density=np.array([2064., 2625., 2544., 2813., 2368., 2168., 1954.])
T43_e_temp=np.array([0.0348, 0.0371, 0.0295, 0.0269, 0.0298, 0.0317, 0.0323])
T43 = [T43_altitude, T43_latitude, T43_loctime, T43_density, T43_e_density, T43_e_temp]

T46_altitude=np.array([1213,1143,1109,1147,1147])
T46_latitude=np.array([-20,-15,-10,0,0])
T46_loctime=np.array([172,175,178,-175,-175])
T46_density=np.array([0.1,2.7,3.8,1.0,1.2])
T46_e_density=np.array([889., 785., 760.,   0., 326.])
T46_e_temp=np.array([0.0437, 0.0433, 0.0178,    np.nan,    np.nan])
T46 = [T46_altitude, T46_latitude, T46_loctime, T46_density, T46_e_density, T46_e_temp]

T48_altitude=np.array([1347,1226,1108,1023,973,973,961,986,1047])
T48_latitude=np.array([-19,-15,-10,-5,0,0,6,12,18])
T48_loctime=np.array([-11,-14,-16,-19,-22,-22,-25,-27,-30])
T48_density=np.array([3.1,17.3,28.2,26.5,24.0,24.9,27.7,19.1,5.6])
T48_e_density=np.array([1438., 2399., 2558., 2541., 2453., 2367., 2448., 2237., 2037.])
T48_e_temp=np.array([0.0774, 0.0608, 0.0447, 0.0419, 0.0408, 0.0386, 0.0408, 0.0488, 0.0727])
T48 = [T48_altitude, T48_latitude, T48_loctime, T48_density, T48_e_density, T48_e_temp]

T49_altitude=np.array([1251,1129,1129,1041,987,970,991,1048])
T49_latitude=np.array([-67,-67,-67,-66,-65,-64,-62,-54])
T49_loctime=np.array([-73,-75,-75,-78,-80,-82,-85,-86])
T49_density=np.array([5.9,15.5,6.6,5.2,4.7,3.7,5.3,7.4])
T49_e_density=np.array([1752., 2139., 1661., 1104., 1139., 1185., 1619., 2153.])
T49_e_temp=np.array([0.0555, 0.0287, 0.0285, 0.0305, 0.0274, 0.047 , 0.032 , 0.0413])
T49 = [T49_altitude, T49_latitude, T49_loctime, T49_density, T49_e_density, T49_e_temp]

T50_altitude=np.array([1250,1250,1127,1038,984,966,986,986,1043])
T50_latitude=np.array([-51,-51,-46,-39,-32,-25,-18,-18,-10])
T50_loctime=np.array([-140,-140,-140,-140,-140,-140,-141,-141,-141])
T50_density=np.array([0.6,1.7,8.6,14.3,11.4,11.5,7.2,5.7,1.2])
T50_e_density=np.array([ 712.,  903.,  952.,  689.,  640.,  581., 1023.,  990.,  736.])
T50_e_temp=np.array([0.0255, 0.0247, 0.0096, 0.0198, 0.0187, 0.0119, 0.0193, 0.0142, 0.0456])
T50 = [T50_altitude, T50_latitude, T50_loctime, T50_density, T50_e_density, T50_e_temp]

T51_altitude=np.array([1355,1211,1097,1016,971,964,993,1060])
T51_latitude=np.array([-65,-63,-60,-55,-43,-7,54,79])
T51_loctime=np.array([-77,-78,-80,-81,-83,-84,-86,-87])
T51_density=np.array([3.9,18.6,15.7,12.6,11.2,9.7,13.6,7.3])
T51_e_density=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
T51_e_temp=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
T51 = [T51_altitude, T51_latitude, T51_loctime, T51_density, T51_e_density, T51_e_temp]


T65_altitude=np.array([1414,1288,1189,1120,1081,1081,1075,1101])
T65_latitude=np.array([-55,-61,-68,-74,-79,-79,-85,-89])
T65_loctime=np.array([-118,-116,-113,-109,-105,-105,-100,-93])
T65_density=np.array([0.1,0.3,0.7,13.6,2.5,3.5,3.2,1.8])
T65_e_density=np.array([ 475.,  579.,  719.,  759., 1035., 1302., 1417., 1387.])
T65_e_temp=np.array([0.0506, 0.0436, 0.0354, 0.0343, 0.0372, 0.0404, 0.0421, 0.0599])
T65 = [T65_altitude, T65_latitude, T65_loctime, T65_density, T65_e_density, T65_e_temp]

T71_altitude=np.array([1372,1240,1135,1059,1015,1015,1004,1025,1078])
T71_latitude=np.array([-50,-63,-76,-87,-82,-82,-74,-68,-62])
T71_loctime=np.array([105,101,97,92,87,87,81,75,69])
T71_density=np.array([1.1,3.6,7.1,7.3,7.2,9.6,14.1,12.7,4.9])
T71_e_density=np.array([ 977., 1268., 1250., 1269., 1278., 1684., 2133., 2463., 1705.])
T71_e_temp=np.array([0.0597, 0.0414, 0.029 , 0.0302, 0.028 , 0.0356,    np.nan, 0.0438, 0.0382])
T71 = [T71_altitude, T71_latitude, T71_loctime, T71_density, T71_e_density, T71_e_temp]

T83_altitude=np.array([1241,1125,1037,980,954,961,961,1001])
T83_latitude=np.array([84,89,82,75,67,60,60,54])
T83_loctime=np.array([-93,-85,-77,-69,-61,-54,-54,-48])
T83_density=np.array([11.0,12.0,7.5,6.7,7.2,15.4,18.3,12.1])
T83_e_density=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
T83_e_temp=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
T83 = [T83_altitude, T83_latitude, T83_loctime, T83_density, T83_e_density, T83_e_temp]

flybys = [TA, T16, T17, T18, T19, T21, T23, T25, T26, T27, T28, T29, T30, T32, T36, T39, T40, T41, T42, T43, T46, T48, T49, T50, T51, T65, T71, T83]
names = ['TA', 'T16', 'T17', 'T18', 'T19', 'T21', 'T23', 'T25', 'T26', 'T27', 'T28', 'T29', 'T30', 'T32', 'T36', 'T39', 'T40', 'T41', 'T42', 'T43', 'T46', 'T48', 'T49', 'T50', 'T51', 'T65', 'T71', 'T83']


layer1 = altitude_layer([], [], [], [])
layer2 = altitude_layer([], [], [], [])
layer3 = altitude_layer([], [], [], [])
layer4 = altitude_layer([], [], [], [])
layer5 = altitude_layer([], [], [], [])

for i in range(0, len(flybys)):
    for j in range(0, len(flybys[i][0])):
        if(flybys[i][0][j]>950 and flybys[i][0][j]<=1050):
            layer1.altitude = np.append(layer1.altitude, int(flybys[i][0][j]))
            layer1.latitude = np.append(layer1.latitude, int(flybys[i][1][j]))
            layer1.local_time = np.append(layer1.local_time, int(flybys[i][2][j]))
            layer1.density = np.append(layer1.density, int(flybys[i][3][j]))
            
        if(flybys[i][0][j]>1050 and flybys[i][0][j]<=1150):
            layer2.altitude = np.append(layer2.altitude, int(flybys[i][0][j]))
            layer2.latitude = np.append(layer2.latitude, int(flybys[i][1][j]))
            layer2.local_time = np.append(layer2.local_time, int(flybys[i][2][j]))
            layer2.density = np.append(layer2.density, int(flybys[i][3][j]))
            
        if(flybys[i][0][j]>1150 and flybys[i][0][j]<=1250):
            layer3.altitude = np.append(layer3.altitude, int(flybys[i][0][j]))
            layer3.latitude = np.append(layer3.latitude, int(flybys[i][1][j]))
            layer3.local_time = np.append(layer3.local_time, int(flybys[i][2][j]))
            layer3.density = np.append(layer3.density, int(flybys[i][3][j]))
            
        if(flybys[i][0][j]>1250 and flybys[i][0][j]<=1350):
            layer4.altitude = np.append(layer4.altitude, int(flybys[i][0][j]))
            layer4.latitude = np.append(layer4.latitude, int(flybys[i][1][j]))
            layer4.local_time = np.append(layer4.local_time, int(flybys[i][2][j]))
            layer4.density = np.append(layer4.density, int(flybys[i][3][j]))
        
        if(flybys[i][0][j]>1350):
            layer5.altitude = np.append(layer5.altitude, int(flybys[i][0][j]))
            layer5.latitude = np.append(layer5.latitude, int(flybys[i][1][j]))
            layer5.local_time = np.append(layer5.local_time, int(flybys[i][2][j]))
            layer5.density = np.append(layer5.density, int(flybys[i][3][j]))
                
    
plt.subplot(428)
plt.scatter(layer1.local_time, layer1.latitude,linewidth=3.0, c=layer1.density*0.25, vmin=0, vmax=10, s=100, cmap='plasma')
plt.title('Altitude range 950-1050 km', fontsize=19.0)
plt.ylim(-90,90)
plt.ylabel('Latitude (\xb0N)', fontsize=19.0)
plt.xlim(-180,180)
plt.xlabel('Titan Local Time', fontsize=19.0)
plt.xticks([-180, -120, -60, 0, 60, 120, 180], ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=16.0)
plt.yticks([-90, -60, -30, 0, 30, 60, 90], ['-90', '-60', '-30', '0', '30', '60', '90'], fontsize=16.0)
cb = plt.colorbar()
cb.set_label(label='Normalised density ($cm^{-3} \cdot \u03B5$)', fontsize=19.0)
cb.ax.tick_params(labelsize=16.0)
plt.text(-210, 105, 'h)', fontsize=25, fontweight='bold', va='top', ha='right')
plt.text(-168, 70, '$C_3N^-$ / $C_4H^-$', bbox=dict(boxstyle="round", facecolor='none'), fontsize=20.0)

plt.plot([-90, -90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([90, 90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([-138, -138], [-90, 90], c='lightgray', linestyle='dotted')
plt.plot([138, 138], [-90, 90], c='lightgray', linestyle='dotted')

plt.subplot(426)
plt.scatter(layer2.local_time, layer2.latitude,linewidth=3.0, c=layer2.density*0.25, vmin=0, vmax=10, s=100, cmap='plasma')
plt.title('Altitude range 1050-1150 km', fontsize=19.0)
plt.ylim(-90,90)
plt.ylabel('Latitude (\xb0N)', fontsize=19.0)
plt.xlim(-180,180)
#plt.xlabel('Titan Local Time', fontsize=19.0)
plt.xticks([-180, -120, -60, 0, 60, 120, 180], ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=16.0)
plt.yticks([-90, -60, -30, 0, 30, 60, 90], ['-90', '-60', '-30', '0', '30', '60', '90'], fontsize=16.0)
cb = plt.colorbar()
cb.set_label(label='Normalised density ($cm^{-3} \cdot \u03B5$)', fontsize=19.0)
cb.ax.tick_params(labelsize=16.0)
plt.text(-210, 105, 'g)', fontsize=25, fontweight='bold', va='top', ha='right')
plt.text(-168, 70, '$C_3N^-$ / $C_4H^-$', bbox=dict(boxstyle="round", facecolor='none'), fontsize=20.0)

plt.plot([-90, -90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([90, 90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([-138, -138], [-90, 90], c='lightgray', linestyle='dotted')
plt.plot([138, 138], [-90, 90], c='lightgray', linestyle='dotted')

plt.subplot(424)
plt.scatter(layer3.local_time, layer3.latitude,linewidth=3.0, c=layer3.density*0.25, vmin=0, vmax=10, s=100, cmap='plasma')
plt.title('Altitude range 1150-1250 km', fontsize=19.0)
plt.ylim(-90,90)
plt.ylabel('Latitude (\xb0N)', fontsize=19.0)
plt.xlim(-180,180)
#plt.xlabel('Titan Local Time', fontsize=19.0)
plt.xticks([-180, -120, -60, 0, 60, 120, 180], ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=16.0)
plt.yticks([-90, -60, -30, 0, 30, 60, 90], ['-90', '-60', '-30', '0', '30', '60', '90'], fontsize=16.0)
cb = plt.colorbar()
cb.set_label(label='Normalised density ($cm^{-3} \cdot \u03B5$)', fontsize=19.0)
cb.ax.tick_params(labelsize=16.0)
plt.text(-210, 105, 'f)', fontsize=25, fontweight='bold', va='top', ha='right')
plt.text(-168, 70, '$C_3N^-$ / $C_4H^-$', bbox=dict(boxstyle="round", facecolor='none'), fontsize=20.0)

plt.plot([-90, -90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([90, 90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([-138, -138], [-90, 90], c='lightgray', linestyle='dotted')
plt.plot([138, 138], [-90, 90], c='lightgray', linestyle='dotted')

plt.subplot(422)
plt.scatter(layer4.local_time, layer4.latitude,linewidth=3.0, c=layer4.density*0.25, vmin=0, vmax=10, s=100, cmap='plasma')
plt.title('Altitude range 1250-1350 km', fontsize=19.0)
plt.ylim(-90,90)
plt.ylabel('Latitude (\xb0N)', fontsize=19.0)
plt.xlim(-180,180)
#plt.xlabel('Titan Local Time', fontsize=19.0)
plt.xticks([-180, -120, -60, 0, 60, 120, 180], ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=16.0)
plt.yticks([-90, -60, -30, 0, 30, 60, 90], ['-90', '-60', '-30', '0', '30', '60', '90'], fontsize=16.0)
cb = plt.colorbar()
cb.set_label(label='Normalised density ($cm^{-3} \cdot \u03B5$)', fontsize=19.0)
cb.ax.tick_params(labelsize=16.0)
plt.text(-168, 70, '$C_3N^-$ / $C_4H^-$', bbox=dict(boxstyle="round", facecolor='none'), fontsize=20.0)
plt.text(-210, 105, 'e)', fontsize=25, fontweight='bold', va='top', ha='right')

plt.plot([-90, -90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([90, 90], [-90, 90], c='lightgray', linestyle='dashed')
plt.plot([-138, -138], [-90, 90], c='lightgray', linestyle='dotted')
plt.plot([138, 138], [-90, 90], c='lightgray', linestyle='dotted')

plt.subplots_adjust(wspace=0.1, hspace=0.2)



plt.savefig('Negative ion variation - species 1 and 2', bbox_inches= None, pad_inches = 0)
