# SRDEM evaluation.
The goal is to evaluate the DEM enhancement with Super Resolution algorithms. To do so, we compare super resolved DEM with some reference (ex. HRDEM downsampled to the same resolution). The data source are two dems.
Data:
DEM_1: srdem: Super Resolution Algorithm Output (ex. at 8m resolution). 
DEM_2: Refference DEM(ex. HRDEM downsamplet at 8m resolution).


Goals: Report some geomorphological measurements to compare the two DEMs. 

The similarity between the DEMs will be evaluated through statistics summary and visual inspection. 

Measurements of comparison (description): 

- Elevation, slope and Flow Accumulation comparison:
        Statistics' summary , compute Min, Man, Mean, STD, Mode and the NoNaNCont. Compare the histogram shapes. 

- Filled area comparison: 
        Fill the dems depressions and pits with WhangAndLiu algorithms. Compute the difference between the original dem <ex. srdem> and the filled dem <ex. srdem_filled>. Compute the percentage of transformed cells on each dem raster. 

- Flow direction:
        The flow direction map (map with prefix: ...d8Pointer.tif, )is used for local comparison. Zooming in the maps, one can evaluate the differences of directions in areas of interest. 

- River network visual comparison:
        Compute strahler order (up to 5th order) and main streams (up to 3rd order). For this, thresholds are setting for 5th order and 
        3rd order (values can be improved). 
        Create maps (Vector) with overlaps of both networks for visual inspection.()

- profile comparison: IMPLEMENTATION IN PROGRESS
        This comparison consists of taking random profiles across the dems and printing it together to visualize the differences. A number of random lines can be set to be drowned through the dems to compute the profiles. As many images as drowned lines will be produced and saved to the working directory. 

Procedure:
 - Fill the configuration file at .\hydraConfig\mainConfig.yaml,  with the path to the dems and the path to the layout (.. path.png). The layout will print an overlap of the extracted river networks from each dem.  
 - Call the main function. Into the main function reportSResDEMComparison() is called. This function automate all calculations and returns a map at every step. One has a choice of erasing intermediary results through by keeping emptyGarbage:bool=True (default). Also, there is the possibility to personalize which maps to keep or erase, un-comment the lines to this purpose into the reportSResDEMComparison() function.  

 Results: Results are maps and images that will be stored in the DEM_1 directory. Therefore, we suggest placing both dems to be compared, in an independent file to keep all results and data source together. 




       

