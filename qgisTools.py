import os
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
    layer_name = file_path.split('/')[-1].split('.')[0]
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
        mapItem.setRect(0,0,150,0)
        mapItem.setReferencePoint(QgsLayoutItem.UpperLeft)
        # mapItem.attemptResize(QgsLayoutSize(145,2))
        mapItem.attemptMove(QgsLayoutPoint(15,15),useReferencePoint=True)
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

    # ## Add a frame item to the layout
    multiframe2 = QgsLayoutItemHtml(layout)
    multiframe2.setHtml('mf2')
    frame_item = QgsLayoutFrame(layout,multiframe2)
    frame_item.setRect(20,20, page.pageSize().width()*0.60,page.pageSize().height()*0.90)
    frame_item.setFrameEnabled(True)
    frame_item.setFrameStrokeWidth(QgsLayoutMeasurement(2, QgsUnitTypes.LayoutMillimeters))
    frame_item.setFrameStrokeColor(QColor(0, 0, 0))
    frame_item.setReferencePoint(QgsLayoutItem.UpperLeft)
    frame_item.attemptMove(QgsLayoutPoint(page.pageSize().width()*0.05,page.pageSize().height()*0.05),useReferencePoint=True)
    layout.addLayoutItem(frame_item)

    # Add a legend item to the layout
    legend_item = QgsLayoutItemLegend(layout)
    legend_item.setReferencePoint(QgsLayoutItem.UpperRight)
    legend_item.attemptResize(QgsLayoutSize(50,50))
    legend_item.attemptMove(QgsLayoutPoint(page.pageSize().width()*0.85,page.pageSize().height()*0.01),useReferencePoint=True)
    layout.addLayoutItem(legend_item)

    return layout



