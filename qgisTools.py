import pathlib
import qgis 
from qgis.gui import QgsMapCanvas, QgsLayerTreeMapCanvasBridge
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsApplication,
    QgsVectorLayer,
    QgsProject,
    QgsLayout,
    QgsLayoutItem,
    QgsLayoutItemLegend,
    QgsLayoutItemHtml,
    QgsLayoutItemMap,
    QgsLayoutFrame,
    QgsLayoutItemPage,
    QgsUnitTypes,
    QgsRectangle,
    QgsCoordinateReferenceSystem,
    QgsLayoutMeasurement,
    QgsLayoutExporter,
    QgsLayoutSize,
    QgsPrintLayout, 
    QgsLayoutPoint
)

def import_vector_layer(file_path):
    layer_name,_ = splitFilenameAndExtention(file_path)
    layer = QgsVectorLayer(file_path, layer_name, "ogr")
    if not layer.isValid():
        print("Layer failed to load!")
    else:
        QgsProject.instance().addMapLayer(layer)
    return layer

def overlap_vectors(vector_file1, vector_file2, outPath):
    
    # Initialize QGIS Application
    qgs = QgsApplication([], False)
    qgs.initQgis()

    # Load the vector layers
    vector_layer1 = import_vector_layer(vector_file1)
    vector_layer2 = import_vector_layer(vector_file2)
    extent = vector_layer1.extent()
    print(f"Sheck-in extent in overlapLayers: {extent}")
    
    # Add the vector layers to the project
    project = QgsProject.instance()
    project.addMapLayer(vector_layer1)
    project.addMapLayer(vector_layer2)
    
    # Create a new layout and add a map item to it
    layout = createCustomLayout(project=project)
    layout_pages = layout.pageCollection()
    first_layout_page = layout_pages.page(0)
    print(first_layout_page.boundingRect())

    for mapItem in project.mapLayers():
        mapItem = QgsLayoutItemMap(layout)
        mapItem.setRect(0,0,200,0)
        mapItem.setReferencePoint(QgsLayoutItem.UpperLeft)
        # mapItem.attemptResize(QgsLayoutSize(145,2))
        mapItem.attemptMove(QgsLayoutPoint(20,35),useReferencePoint=True)
        layout.addLayoutItem(mapItem)
        mapItem.setExtent(extent)
        
    # Export the layout as an image in PNG format
    exporter = QgsLayoutExporter(layout)
    exporter.exportToImage(outPath, QgsLayoutExporter.ImageExportSettings())

    # Clean up
    del exporter
    del layout
    del qgs


def createCustomLayout(project:QgsProject, name:str = 'My Layout'):
    # Create a new layout
    manager = project.layoutManager()
    
    layout = QgsLayout(project)
    layout.initializeDefaults()
    layout = QgsPrintLayout(project)
    layout.setName(name)
    manager.addLayout(layout)

    # Create Layout background page
    page = QgsLayoutItemPage(layout)
    page.setPageSize('A4', QgsLayoutItemPage.Orientation.Landscape)
    layout.pageCollection().addPage(page)

    # # ## Add a frame item to the layout
    # multiframe2 = QgsLayoutItemHtml(layout)
    # multiframe2.setHtml('mf2')
    # frame_item = QgsLayoutFrame(layout,multiframe2)
    # frame_item.setRect(20,20, page.pageSize().width()*0.60,page.pageSize().height()*0.90)
    # frame_item.setFrameEnabled(True)
    # frame_item.setFrameStrokeWidth(QgsLayoutMeasurement(2, QgsUnitTypes.LayoutMillimeters))
    # frame_item.setFrameStrokeColor(QColor(0, 0, 0))
    # frame_item.setReferencePoint(QgsLayoutItem.UpperLeft)
    # frame_item.attemptMove(QgsLayoutPoint(page.pageSize().width()*0.05,page.pageSize().height()*0.05),useReferencePoint=True)
    # layout.addLayoutItem(frame_item)

    # Add a legend item to the layout
    legend_item = QgsLayoutItemLegend(layout)
    legend_item.setReferencePoint(QgsLayoutItem.UpperRight)
    legend_item.attemptResize(QgsLayoutSize(50,50))
    legend_item.attemptMove(QgsLayoutPoint(page.pageSize().width()*0.70,page.pageSize().height()*0.01),useReferencePoint=True)
    layout.addLayoutItem(legend_item)

    return layout


###  TO BE Tested

# from qgis.core import QgsGeometry, QgsPointXY, QgsVectorLayer, QgsFeature, QgsSymbolLayerRegistry, QgsLineSymbolV2, QgsSimpleLineSymbolLayerV2, QgsMarkerSymbolV2, QgsFillSymbolV2, QgsRendererCategoryV2, QgsCategorizedSymbolRendererV2
# from PyQt5.QtCore import QSize
# from PyQt5.QtGui import QColor
# from PyQt5.QtSvg import QSvgRenderer
# from PyQt5.QtWidgets import QApplication

# def create_image(vector1, vector2):
#     # Create a new vector layer
#     layer = QgsVectorLayer("LineString", "My Layer", "memory")
#     pr = layer.dataProvider()

#     # Add the first vector to the layer
#     feature1 = QgsFeature()
#     feature1.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(vector1[0], vector1[1]), QgsPointXY(vector1[2], vector1[3])]))
#     pr.addFeatures([feature1])

#     # Add the second vector to the layer
#     feature2 = QgsFeature()
#     feature2.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(vector2[0], vector2[1]), QgsPointXY(vector2[2], vector2[3])]))
#     pr.addFeatures([feature2])

#     # Set the symbol for the first vector to red
#     line_symbol_layer_red = QgsSimpleLineSymbolLayerV2()
#     line_symbol_layer_red.setColor(QColor(255, 0, 0))
#     line_symbol_layer_red.setWidth(0.5)
#     line_symbol_red = QgsLineSymbolV2([line_symbol_layer_red])
#     renderer_red = QgsCategorizedSymbolRendererV2('id', [QgsRendererCategoryV2('1', line_symbol_red)])
#     layer.setRendererV2(renderer_red)

#     # Set the symbol for the second vector to blue
#     line_symbol_layer_blue = QgsSimpleLineSymbolLayerV2()
#     line_symbol_layer_blue.setColor(QColor(0, 0, 255))
#     line_symbol_layer_blue.setWidth(0.5)
#     line_symbol_blue = QgsLineSymbolV2([line_symbol_layer_blue])
#     renderer_blue = QgsCategorizedSymbolRendererV2('id', [QgsRendererCategoryV2('1', line_symbol_blue)])
#     layer.setRendererV2(renderer_blue)

#     # Create a PNG image of the layout
#     layout = QgsPrintLayout(QgsProject.instance())
#     layout.initializeDefaults()
#     layout.setName("My Layout")
    
#     manager = QgsProject.instance().layoutManager()
#     manager.addLayout(layout)

#     map_item = layout.addItem(QgsLayoutItemMap(layout))
    
#     map_item.setRect(20, 20, 200, 200)
    
#     map_item.setLayers([layer])
    
#     exporter = QImage(QSize(200, 200), QImage.Format_ARGB32_Premultiplied)
    
#     p = QPainter(exporter)
    
#     map_item.render(p)
    
#     p.end()

#     exporter.save("output.png", "PNG")

# # Example usage of create_image function
# vector1 = [0, 0, 10, 10]
# vector2 = [10, 0, 0, 10]
# create_image(vector1, vector2)

def splitFilenameAndExtention(file_path):
    '''
    pathlib.Path Options: 
    '''
    fpath = pathlib.Path(file_path)
    extention = fpath.suffix
    name = fpath.stem
    return name, extention 