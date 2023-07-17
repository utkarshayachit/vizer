from paraview.util.vtkAlgorithm import *
from paraview import vtk
from vtkmodules.vtkFiltersSources import vtkOutlineSource
from vtkmodules.vtkFiltersCore import vtkAppendPolyData


def compute_bounds(image, voi):
    bbox = vtk.vtkBoundingBox()
    for i in range(8):
        p = [0, 0, 0]
        p[0] = voi[0] if i & 1 else voi[1]
        p[1] = voi[2] if i & 2 else voi[3]
        p[2] = voi[4] if i & 4 else voi[5]
        bbox.AddPoint(image.GetPoint(image.ComputePointId(p)))
    bds = [0] * 6
    bbox.GetBounds(bds)
    return bds

@smproxy.filter()
@smproperty.input(name="Input")
@smdomain.datatype(dataTypes=["vtkImageData"], composite_data_supported=False)
class ImageOutlineFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=1,
            nOutputPorts=1,
            outputType="vtkPolyData",
        )
        self._slices =[[]] * 3

    @smproperty.xml("""
        <IntVectorProperty name="XSlices"
            label="X Slices"
            command="SetXSlices"
            repeatable="1"
            number_of_elements_per_command="1"
            set_number_command="SetNumberOfXSlices"
            use_index="1" />
    """)
    def SetXSlices(self, idx, x):
        self._slices[0][idx] = x
        self.Modified()

    def SetNumberOfXSlices(self, n):
        self._slices[0] = [0] * n
        self.Modified()

    @smproperty.xml("""
        <IntVectorProperty name="YSlices"
            label="Y Slices"
            command="SetYSlices"
            repeatable="1"
            number_of_elements_per_command="1"
            set_number_command="SetNumberOfYSlices"
            use_index="1" />
    """)
    def SetYSlices(self, idx, x):
        self._slices[1][idx] = x
        self.Modified()

    def SetNumberOfYSlices(self, n):
        self._slices[1] = [0] * n
        self.Modified()

    @smproperty.xml("""
        <IntVectorProperty name="ZSlices"
            label="Z Slices"
            command="SetZSlices"
            repeatable="1"
            number_of_elements_per_command="1"
            set_number_command="SetNumberOfZSlices"
            use_index="1" />
    """)
    def SetZSlices(self, idx, x):
        self._slices[2][idx] = x
        self.Modified()

    def SetNumberOfZSlices(self, n):
        self._slices[2] = [0] * n
        self.Modified()

    def FillInputPortInformation(self, port, info):
        info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkImageData")
        return 1

    def RequestData(self, request, inInfoVec, outInfoVec):
        input = vtk.vtkImageData.GetData(inInfoVec[0])
        output = vtk.vtkPolyData.GetData(outInfoVec)
        # print("slices", self._slices)

        appender = vtkAppendPolyData()
        ext = input.GetExtent()

        colors = [ [1., 0., 0.], [1., 1., 0.],  [0., 1., 0.] ]
        for axis in range(3):
            scalars = vtk.vtkUnsignedCharArray()
            scalars.SetNumberOfComponents(3)
            scalars.SetName("colors")
            scalars.SetNumberOfTuples(8)
            scalars.FillComponent(0, colors[axis][0] * 255)
            scalars.FillComponent(1, colors[axis][1] * 255)
            scalars.FillComponent(2, colors[axis][2] * 255)
            for slice in set(self._slices[axis]):
                voi = list(ext)
                if slice < ext[axis*2] or slice > ext[axis*2+1]:
                    continue
                voi[axis*2] = voi[axis*2+1] = slice
                bounds = compute_bounds(input, voi)
                outline = vtkOutlineSource()
                outline.SetBounds(bounds)
                outline.Update()
                data = outline.GetOutput()
                data.GetPointData().SetScalars(scalars)
                appender.AddInputData(data)
        if appender.GetNumberOfInputConnections(0) > 0:
            appender.Update()
            output.ShallowCopy(appender.GetOutput())
        return 1
