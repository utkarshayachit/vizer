from paraview.util.vtkAlgorithm import *
from vtkmodules.vtkAcceleratorsVTKmFilters import vtkmExtractVOI
# from vtkmodules.vtkImagingCore import vtkExtractVOI
from paraview.vtk import *


@smproxy.filter()
@smproperty.input(name="Input")
@smdomain.datatype(dataTypes=["vtkImageData"], composite_data_supported=False)
class ExtractVOI(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=1,
            nOutputPorts=1,
            outputType="vtkImageData",
        )
        # self._filter = vtkExtractVOI()
        self._filter = vtkmExtractVOI()
        self._filter.ForceVTKmOn()
        self._filter.SetIncludeBoundary(0)
        self._filter.SetSampleRate(1, 1, 1)

    @smproperty.intvector(name="VOI", default_values=[0, 0, 0, 0, 0, 0])
    def SetVOI(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self._filter.SetVOI(xmin, xmax, ymin, ymax, zmin, zmax)
        self.Modified()

    def RequestInformation(self, request, inInfoVector, outInfoVector):
        executive = self.GetExecutive()
        inInfo = inInfoVector[0].GetInformationObject(0)
        outInfo = outInfoVector.GetInformationObject(0)
        outInfo.Set(executive.WHOLE_EXTENT(), self._filter.GetVOI(), 6)
        outInfo.CopyEntry(inInfo, vtkImageData.ORIGIN())
        outInfo.CopyEntry(inInfo, vtkImageData.SPACING())
        return 1

    def RequestUpdateExtent(self, request, inInfoVector, outInfoVector):
        executive = self.GetExecutive()
        inInfo = inInfoVector[0].GetInformationObject(0)
        inInfo.Set(executive.UPDATE_EXTENT(), self._filter.GetVOI(), 6)
        return 1
        
    def RequestData(self, request, inInfo, outInfo):
        inputImage = vtkImageData.GetData(inInfo[0], 0)
        outputImage = vtkImageData.GetData(outInfo, 0)

        self._filter.SetInputDataObject(inputImage)
        self._filter.Update()
        outputImage.ShallowCopy(self._filter.GetOutputDataObject(0))
        self._filter.SetInputDataObject(None)
        return 1

         