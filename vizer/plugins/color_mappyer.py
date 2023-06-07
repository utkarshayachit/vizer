from paraview.util.vtkAlgorithm import *
from paraview.vtk import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa

import numpy as np
import color_mappyer

@smproxy.filter()
@smproperty.input(name="Input")
@smdomain.datatype(dataTypes=["vtkImageData"], composite_data_supported=False)
class ColorMappyer(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=1,
            nOutputPorts=1,
            outputType="vtkImageData",
        )
        self._scalars = None
        self._colors = None

        self._scalars_cache = None
        self._colors_cache = None

    @smproperty.xml("""
        <StringVectorProperty name="Scalars"
            command="SetScalars"
            argument_is_array="1">
        </StringVectorProperty>
        """)
    def SetScalars(self, scalars):
        # note: scalars is a list of strings
        self._scalars = scalars
        self._scalars_cache = None
        self.Modified()

    @smproperty.xml("""
        <DoubleVectorProperty name="Colors"
            command="SetColors"
            argument_is_array="1">
        </DoubleVectorProperty>
        """)
    def SetColors(self, colors):
        # note: colors is a list of floats in range (0, 1)
        self._colors = colors
        self._colors_cache = None
        self.Modified()

    def _get_scalars(self, dtype):
        if self._scalars_cache is None or self._scalars_cache.dtype != dtype:
            self._scalars_cache = np.array(self._scalars, dtype=dtype)
        return self._scalars_cache
    
    def _get_colors(self):
        if self._colors_cache is None:
            shape = (len(self._colors)//4, 4,)
            self._colors_cache = (np.array(self._colors, dtype=np.float32).reshape(shape) * 255).astype(np.uint8)
        return self._colors_cache

    def RequestData(self, request, inInfo, outInfo):
        inputImage = vtkImageData.GetData(inInfo[0], 0)
        outputImage = vtkImageData.GetData(outInfo, 0)

        if self._scalars is None or len(self._scalars) == 0 or self._colors is None or len(self._colors) == 0:
            outputImage.ShallowCopy(inputImage)
            return 1

        outputImage.CopyStructure(inputImage) 
        inputData = dsa.WrapDataObject(inputImage)
        outputData = dsa.WrapDataObject(outputImage)

        scalarsArrayName = inputData.PointData.GetScalars().GetName()
        data = inputData.PointData[scalarsArrayName]

        result = color_mappyer.discrete(data, self._get_scalars(data.dtype), self._get_colors())
        # result = result.view(np.uint8).reshape(result.shape + (-1,))
        outputData.PointData.append(result, scalarsArrayName)
        return 1
