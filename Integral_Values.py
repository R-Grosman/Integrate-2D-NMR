"""
==============================================================================
Title        : Integral_Values.py
Description  : Calculate integration of multiple regions in 2D NMR using a
               .xlsx file.
==============================================================================
Author       : Rudi Grosman - University of Liverpool, High Field NMR Facility
Date         : 2021-10-25
Version      : 1.0
==============================================================================
Usage        : 
    python Integral_Values.py
    
    - Example: 
        python Integral_Values.py
==============================================================================
Requirements : 
    - Python version >= 3.9
    - pandas
    - numpy
    - scipy
    - nmrglue

==============================================================================
Notes        : 
    Global Variables:
    1. REGIONS : str : Path to a list of integration regions containing
                         F1 Left, F1 Right, F2 Left, F2 Right, Assignment
    2. DATASET_PATH : str : Path to the dataset to be used
    3. EXPERIMENT_NO : int : Experiment number to be integrated

==============================================================================
License      : GNU General Public License v3.0
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.

==============================================================================
"""

import csv
from pathlib import Path

import nmrglue as ng
import numpy as np
import pandas as pd
from scipy.integrate import dblquad

### Set Global Variables
# Path to a list of integration regions with the columns:
# F1 Left ppm, F1 Right ppm, F2 Left ppm, F2 Right ppm, Assignment
REGIONS = 'Input_regions.xlsx'

# Path of the dataset
DATASET_PATH = Path().home()

# Experiment number as an integer
EXPERIMENT_NO = 1
###

class Spectrum:
    def __init__(self, datapath):
        self.path = datapath
        self.dataset_name = self.path.parent.parent.parent.stem
        self.ExpNo = self.path.parent.parent.stem
        self.procNo = self.path.stem
        self.dic = ng.bruker.read_pdata(self.path)[0]
        self.udic = ng.bruker.guess_udic(self.dic, self.get_data())
        self.ucF1 = ng.fileiobase.uc_from_udic(self.udic, 0)
        self.ucF2 = ng.fileiobase.uc_from_udic(self.udic, 1)
        self.Peaks = []

    def get_data(self):
        return np.array(ng.bruker.read_pdata(self.path)[1], dtype='float32')

    def addPeak(self, F1ppmL, F1ppmR, F2ppmL, F2ppmR, assignment=None):
        self.Peaks.append(Peak2D(self, F1ppmL, F1ppmR, F2ppmL, F2ppmR,
                                 assignment))

    def findPeak(self, assignment_):
        assignlist = [peak.assignment for peak in self.Peaks]
        try:
            match = assignlist.index(assignment_)
            return self.Peaks[match]
        except ValueError:
            print('No matching assignment found in peaks')
            return None

class Peak2D:
    def __init__(self, SpecObj, F1ppmL, F1ppmR, F2ppmL, F2ppmR, assignment):

        self.F1ppmL = float(F1ppmL)
        self.F1ppmR = float(F1ppmR)
        self.F2ppmL = float(F2ppmL)
        self.F2ppmR = float(F2ppmR)
        self.assignment = str(assignment)
        self.peakMatrix, self.peakVolume, self.F1PeakScale, self.F2PeakScale = self.ExtractPeak(
            SpecObj)

        self.BaseCorCoefs, self.BaseCorFunction, self.BaseCorString = self.CalcCoefs()
        self.BaseCorVolume, self.BaseCorError = self.CalcCorVol()

    def ExtractPeak(self, SpecObj):
        F1ppmR_Ind = SpecObj.ucF1(self.F1ppmR, "ppm")
        F1ppmL_Ind = SpecObj.ucF1(self.F1ppmL, "ppm")
        F2ppmR_Ind = SpecObj.ucF2(self.F2ppmR, "ppm")
        F2ppmL_Ind = SpecObj.ucF2(self.F2ppmL, "ppm")

        dataSub = SpecObj.get_data()[F1ppmL_Ind:F1ppmR_Ind + 1,
                  F2ppmL_Ind:F2ppmR_Ind + 1]
        PeakVol = dataSub.sum()
        F1PeakScale = SpecObj.ucF1.ppm_scale()[F1ppmL_Ind:F1ppmR_Ind + 1]
        F2PeakScale = SpecObj.ucF2.ppm_scale()[F2ppmL_Ind:F2ppmR_Ind + 1]

        if len(dataSub) == 0:
            print(self.assignment, "Peak matrix empty")

        return dataSub, PeakVol, F1PeakScale, F2PeakScale
    
    def CalcCoefs(self):
        f1ppm = [self.F1ppmR, self.F1ppmL]
        f2ppm = [self.F2ppmR, self.F2ppmL]
        corners = self.peakMatrix[::self.peakMatrix.shape[0] - 1,
                  ::self.peakMatrix.shape[1] - 1]
        Mat = np.array([[1, f2ppm[1], f1ppm[0], f2ppm[1] * f1ppm[0]],
                        [1, f2ppm[0], f1ppm[0], f2ppm[0] * f1ppm[0]],
                        [1, f2ppm[1], f1ppm[1], f2ppm[1] * f1ppm[1]],
                        [1, f2ppm[0], f1ppm[1], f2ppm[0] * f1ppm[1]]])
        coefs = np.linalg.solve(Mat, corners.flatten())
        coefs = {'a0': coefs[0], 
                 'a1': coefs[1],
                 'a2': coefs[2],
                 'a3': coefs[3]
                 }
        BCF = lambda y, x: coefs['a0'] \
                           + coefs['a1'] * x \
                           + coefs['a2'] * y \
                           + coefs['a3'] * x * y
        BCFstring = 'f(x,y) = {:.2f} + {:.2f} * x + {:.2f} * y + {:.2f} * x ' \
                    '* y'.format(coefs['a0'], coefs['a1'], coefs['a2'], coefs['a3'])
        return coefs, BCF, BCFstring
        

    def CalcCorVol(self):
        Volume = dblquad(self.BaseCorFunction, self.F2ppmR, self.F2ppmL,
                         lambda x: self.F1ppmR, lambda x: self.F1ppmL)
        return Volume[0], Volume[1]

def setenv(specno, dataset_path=DATASET_PATH, regions=REGIONS):
    regions = pd.read_excel(REGIONS)
    tmpspec = Spectrum(Path(dataset_path , str(specno) , 'pdata/1'))
    for _, row in regions.iterrows():
        tmpspec.addPeak(*row)
    return tmpspec

def getintegs(fname, spc):
  tmplist = []
  header = ['Assignment', 'Integration','CorrectedIntegration']
  for peak in spc.Peaks:
    tmplist.append([peak.assignment,peak.peakVolume, peak.peakVolume - peak.BaseCorVolume])
  
  with(open(fname, 'w+', newline='')) as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(tmplist)


if __name__ == '__main__':
    tmpspec = setenv(EXPERIMENT_NO)
    getintegs(f'Integrations_{EXPERIMENT_NO}.csv', tmpspec)
    print('DONE!')
