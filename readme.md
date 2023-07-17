# vizer: a dataset viewer for the Web

A simple viewer for raw volumes and other datasets supported by [ParaView]. This is a Python-based
application that uses [trame], a Python framework to create interactive web applications
and [ParaView], a scientific data analaysis and visualization framework.

This application is not intended to be a complete solution but rather a starting point for
building a custom viewer for your data. It is also a good example of how to use [trame] to
create a web application together with ParaView. The application was initially developed
as a prototype to demonstrate how an interactive web visualization application could be
deployed on Azure. Since the initial demonstrative post [here](https://techcommunity.microsoft.com/t5/azure-high-performance-computing/interactive-web-based-3d-visualization-of-large-scientific/ba-p/3686390),
the application has been extended to support an eclectic set of features as requested by
assorted users.

## License

Copyright (c) Microsoft Corporation. All rights reserved.

This project is licensed under the terms of the [MIT license](LICENSE).

[ParaView]: https://www.paraview.org/
[trame]: https://kitware.github.io/trame/
