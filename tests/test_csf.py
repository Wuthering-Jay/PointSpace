# coding: utf-8
import laspy
import CSF
import numpy as np

inFile = laspy.read(r"E:\data\DALES\dales_las\train\5080_54435.las") # read a las file
points = inFile.points
xyz = np.vstack((inFile.x, inFile.y, inFile.z)).transpose() # extract x, y, z and put into a list

csf = CSF.CSF()

# prameter settings
csf.params.bSloopSmooth = False
csf.params.cloth_resolution = 5
# more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/

csf.setPointCloud(xyz)
ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
csf.do_filtering(ground, non_ground) # do actual filtering.

outFile = laspy.LasData(inFile.header)
outFile.points = points[np.array(ground)] # extract ground points, and save it to a las file.
outFile.write(r"E:\data\DALES\dales_las\csf\5080_54435.las")