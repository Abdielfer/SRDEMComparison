import qgisTools as QT
import os, ntpath
import glob
import pathlib
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show_hist
from datetime import datetime
from whitebox.whitebox_tools import WhiteboxTools, default_callback  
from osgeo import gdal,ogr, osr
from osgeo import gdal_array
from osgeo.gdalconst import *
from omegaconf import DictConfig, OmegaConf
import hydra
import logging 


### General applications ##
class timeit(): 
    '''
    to compute execution time do:
    with timeit():
         your code, e.g., 
    '''
    def __enter__(self):
        self.tic = datetime.now()
    def __exit__(self, *args, **kwargs):
        print('runtime: {}'.format(datetime.now() - self.tic))

### Configurations And file management

def makePath(str1,str2):
    return os.path.join(str1,str2)

def removeFile(filePath):
    try:
        os.remove(filePath)
        return True
    except OSError as error:
        print(error)
        print("File path can not be removed")
        return False
    
def clearTransitFolderContent(path:str, filetype = '/*'):
    '''
    NOTE: This well clear dir without removing the parent dir itself. 
    We can replace '*' for an specific condition ei. '.tif' for specific fileType deletion if needed. 
    @Arguments:
    @path: Parent directory path
    @filetype: file type to delete. @default ='/*' delete all files. 
    '''
    files = glob.glob(path + filetype)
    for f in files:
        os.remove(f)
    return True

def createListFromExelColumn(excell_file_location,Sheet_id:str, col_idx:str):  
    '''
    @return: list from <col_id> in <excell_file_location>.
    Argument:
    @excell_file_location: full path file location and name.
    @col_id : number of the desired collumn to extrac info from (Consider index 0 for the first column)
    '''       
    x=[]
    df = pd.ExcelFile(excell_file_location).parse(Sheet_id)
    for i in df[col_idx]:
        x.append(i)
    return x

def splitFilenameAndExtention(file_path):
    '''
    pathlib.Path Options: 
    '''
    fpath = pathlib.Path(file_path)
    extention = fpath.suffix
    name = fpath.stem
    return name, extention 

def replaceExtention(inPath,newExt: str)->os.path :
    '''
    Just remember to add the poin to the new ext -> '.map'
    '''
    dir,fileName = ntpath.split(inPath)
    _,actualExt = ntpath.splitext(fileName)
    return os.path.join(dir,ntpath.basename(inPath).replace(actualExt,newExt))

def get_parenPath_name_ext(filePath):
    '''
    Ex: user/folther/file.txt
    parentPath = pathlib.PurePath('/src/goo/scripts/main.py').parent 
    parentPath => '/src/goo/scripts/'
    parentPath: can be instantiated.
         ex: parentPath[0] => '/src/goo/scripts/'; parentPath[1] => '/src/goo/', etc...
    '''
    parentPath = pathlib.PurePath(filePath).parent
    name, ext = splitFilenameAndExtention(filePath)
    return parentPath, name, ext
  
def addSubstringToName(path, subStr: str, destinyPath = None) -> os.path:
    '''
    @path: Path to the raster to read. 
    @subStr:  String o add at the end of the origial name
    @destinyPath (default = None)
    '''
    parentPath,name,ext= get_parenPath_name_ext(path)
    if destinyPath != None: 
        return os.path.join(destinyPath,(name+subStr+ext))
    else: 
        return os.path.join(parentPath,(name+subStr+ ext))

def replaceName_KeepPathAndExt(path, newName: str) -> os.path:
    '''
    @path: Path to the raster to read. 
    @subStr:  String o add at the end of the origial name
    @destinyPath (default = None)
    '''
    parentPath,_,ext= get_parenPath_name_ext(path)
    return os.path.join(parentPath,(newName+ext))
    
def createCSVFromList(pathToSave: os.path, listData:list):
    '''
    This function create a *.csv file with one line per <lstData> element. 
    @pathToSave: path of *.csv file to be writed with name and extention.
    @listData: list to be writed. 
    '''
    parentPath,name,_ = get_parenPath_name_ext(pathToSave)
    textPath = makePath(parentPath,(name+'.txt'))
    with open(textPath, 'w') as output:
        for line in listData:
            output.write(str(line) + '\n')
    read_file = pd.read_csv (textPath)
    print(f'Creating CSV at {pathToSave}')
    read_file.to_csv (pathToSave, index=None)
    removeFile(textPath)
    return True

def updateDict(dic:dict, args:dict)->dict:
    outDic = dic
    for k in args.keys():
        if k in dic.keys():
            outDic[k]= args[k]
    return outDic

class logg_Manager:
    '''
    This class creates a logger object that writes logs to both a file and the console. 
    @log_name: lLog_name. Logged at the info level by default.
    @log_dict: Dictionary, Sets the <attributes> with <values> in the dictionary. 
    The logger can be customized by modifying the logger.setLevel and formatter attributes.

    The update_logs method takes a dictionary as input and updates the attributes of the class to the values in the dictionary. The method also takes an optional level argument that determines the severity level of the log message. 
    '''
    def __init__(self,logName:str, log_dict:dict = {} ):# log_name, 
        logpath = os.path.join(self.getHydraOutputDri(),'logName')
        self.logger = logging.getLogger('logName')
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler = logging.FileHandler(logpath)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(self.formatter)
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(logging.ERROR)
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stream_handler)
        for key, value in log_dict.items():
            setattr(self, key, value)
            self.logger.info(f'{key} - {value}')
    
    def update_logs(self, log_dict, level=logging.INFO):
        for key, value in log_dict.items():
            setattr(self, key, value)
            if level == logging.DEBUG:
                self.logger.debug(f'{key} - {value}')
            elif level == logging.WARNING:
                self.logger.warning(f'{key} - {value}')
            elif level == logging.ERROR:
                self.logger.error(f'{key} - {value}')
            else:
                self.logger.info(f'{key} - {value}')
    @staticmethod
    def getHydraOutputDri()-> str:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()   
        return hydra_cfg['runtime']['output_dir']

def update_logs(log_dict, level=logging.INFO):
        for key, value in log_dict.items():
            if level == logging.DEBUG:
                logging.debug(f'{key} - {value}')
            elif level == logging.WARNING:
                logging.warning(f'{key} - {value}')
            elif level == logging.ERROR:
                logging.error(f'{key} - {value}')
            else:
                logging.info(f'{key} - {value}')

def saveLogsAsTxt():
    '''
    Save the hydra default logger as txt file. 
    '''
    # Create a logger object
    logger = logging.getLogger()
    # Set the logging level
    logger.setLevel(logging.DEBUG)
    # Create a file handler
    handler = logging.FileHandler('logs.txt')
    # Set the logging format
    formatter = logging.Formatter('%(name)s - %(message)s')
    handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(handler)

###################            
### General GIS ###
###################

def reshape_as_raster(arr):
    '''  
    From GDL
        swap the axes order from (rows, columns, bands) to (bands, rows, columns)
    Parameters
    ----------
    arr : array-like in the image form of (rows, columns, bands)
    return: arr as image in raster order (bands, rows, columns)
    '''
    return np.transpose(arr, [2, 0, 1])

def readRasterAsArry(rasterPath):
   return gdal_array.LoadFile(rasterPath)

def plotRasterHistComparison(DEM1,DEM2,title:str='', ax_x_units:str='', bins:int = 100, addmax= False):
    '''
    

    '''
    _,dem1_Name,_ = get_parenPath_name_ext(DEM1)
    _,dem2_Name,_ = get_parenPath_name_ext(DEM2)
    dem_1_Array = readRasteReplacingNoDataWithNan(DEM1)
    dem_2_Array = readRasteReplacingNoDataWithNan(DEM2)
    
    ## If <bins> is a list, add the maximum value to <bins>.  
    if (addmax and isinstance(bins,list)):
        bins.append(np.maximum(np.nanmax(dem_1_Array),np.nanmax(dem_2_Array)).astype(int))
    
    # Reading raster. 
    dataRechaped_1 = np.reshape(dem_1_Array,(1,-1))
    dataRechaped_2 = np.reshape(dem_2_Array,(1,-1))
    
    # Prepare plot
    fig, ax = plt.subplots(1,sharey=True, tight_layout=True)
    ax.hist([dataRechaped_1[0],dataRechaped_2[0]],
            bins,
            rwidth=0.8)
    ax.legend([f"{dem1_Name}",f"{dem2_Name}"], prop={'size': 8})
    ax.set_title(title)
    ax.set_xlabel(ax_x_units) 
    ax.set_ylabel('Frequency')
   
    if isinstance(bins,list):
        plt.xticks(bins)
        print(bins)
        plt.gca().set_xticklabels([str(i) for i in bins], minor = True)
    
    # plt.show()
    
def reportSResDEMComparison(cfg: DictConfig, emptyGarbage:bool=True):
    '''
    The goal of this function is to create a report of comparison, between two DEMs.
    BOTH DEMs must be in the same folder. ALL outputs will be writen to this folder. You can automaticaly delete all intermediary results of your choice by filling the @emptyGarbage list. 
    
    Inputs: 
    @cfg: DictConfig. Hydra config dictionary containing the path to the DEMs to be compared as <dem_1> and <dem_2>
    @emptyGarbage: bool: default True. Whether to delete or not all the intermediary results. 

    Data:
    The expected data are DEMs. 
        srdem: Super Resolution Algorithm Output. 
        DEM to compare with srdem. 


    Goals: Report some geomorphological measurements to compare the super-Resolution algorithms output with the DEM. 

        The similarity between the DEM and srdem will be evaluated through statistics summary and visual instapection. Since both dems could be not in the same resolution, the comparison is made in terms of percentage and visual inspection. 
        
    Measurements of comparison (description): 

        - Elevation, slope and Flow Accumulation comparison:
                statistics' summary , compute Min, Man, Mean, STD, Mode and the NoNaNCont. Compare the histogram shapes. 
        
        - Filled area comparison: 
                Fill the dems depressions and pits with WhangAndLiu algorithms. Compute the difference between the original dem <ex. srdem> and the filled dem <ex. srdem_filled>. Compute the percentage of transformed cells on each dem raster. 

        - River network visual comparison:
                Compute strahler order (up to 5th order) and main streams (up to 3rd order). For this, thresholds are setting for 5th order and 
                3rd order (values can be improved). 
                Create maps (Vector) with overlaps of both networks for visual inspection.()
    '''
    
    ## Inputs
    dem_1 = cfg['dem_1']
    parentDirDEM_1,dem1_Name,_ = get_parenPath_name_ext(dem_1) 
    dem_2 = cfg['dem_2']
    _,dem2_Name,_ = get_parenPath_name_ext(dem_2) 
    layOutPath = cfg['layOutOutputPath']

    # Collect intermediary results to be deleted at the end.
    garbageList = []
    
    strahOrdThreshold_5th = 30000  # Experimental values 
    strahOrdThreshold_3rd = 300000   # Experimental values 
    
    ## Initialize WhiteBoxTools working directory at the parent directory of dem_1.    
    WbT = WbT_dtmTransformer(parentDirDEM_1)
       
    ## Prepare report logging
    logging.info({"WBTools working dir ": WbT.get_WorkingDir()})

    #______ Elevation statistics before filling : Compute mean, std, mode, max and min. Compare elevation histograms."
    dem_1_ElevStats = computeRaterStats(dem_1)
    dem_2_ElevStats = computeRaterStats(dem_2)
    
       # Log Elevation Stats.
    update_logs({f"{dem1_Name} elevation stats before filling:": dem_1_ElevStats})
    update_logs({f"{dem2_Name} elevation stats before filling: ": dem_2_ElevStats})
        # plot elevation histogram
    plotRasterHistComparison(dem_1,dem_2,title=f' Elevation comparison: {dem1_Name} vs {dem2_Name}', ax_x_units ='Elevation (m)')
    
    #______ Filled area comparison: Compute mean, std, mode, max and min. Compare the histograms."
    #______ Fill the DEMs with WhangAndLiu algorithms from WhiteBoxTools
    dem_1_Filled = WbT.fixNoDataAndfillDTM(dem_1)
    dem_2_Filled = WbT.fixNoDataAndfillDTM(dem_2)

    # garbageList.append(dem_1_Filled) 
    # garbageList.append(dem_2_Filled) 

    # ___ Compute and log percent of transformed areas. 
            ########  dem_1
    dem_1_statement = str("'"+dem_1_Filled+"'"+'-'+"'"+dem_1+"' > 0.05") # Remove some noise because of approximations with -0.05
    dem_1_transformations_path = addSubstringToName(dem_1,'_TransformedArea')
    dem_1_Transformations_binary = WbT.rasterCalculator(dem_1_transformations_path,dem_1_statement)
    dem_1_Transformation_percent = computeRasterValuePercent(dem_1_Transformations_binary)
    update_logs({f"Depressions and pit percent in {dem1_Name} ": dem_1_Transformation_percent})

            ########  dem_2
    dem_2_statement = str("'"+dem_2_Filled+"'"+' - '+"'"+dem_2+"' > 0.05") # Remove some noise because of approximations with -0.05
    dem_2_transformations_path = addSubstringToName(dem_2,'_TransformedArea')
    dem_2_Transformations_binary = WbT.rasterCalculator(dem_2_transformations_path,dem_2_statement)
    dem_2_Transformation_percent = computeRasterValuePercent(dem_2_Transformations_binary)
    update_logs({f"Depressions and pit percent in {dem2_Name} ": dem_2_Transformation_percent})

   ##______ Elevations statistics AFTER filling : Compute mean, std, mode, max and min. Compare elevation histograms."
    dem_1_FilledElevStats = computeRaterStats(dem_1_Filled)
    dem_2_FilledElevStats = computeRaterStats(dem_2_Filled)
        # Log Elevation Stats.
    update_logs({f"{dem1_Name} Filled elevation stats ": dem_1_FilledElevStats})
    update_logs({f"{dem2_Name} Filled elevation stats ": dem_2_FilledElevStats})
        # plot elevation histogram of filled dems.
    plotRasterHistComparison(dem_1_Filled,dem_2_Filled,title=f"Elevation comparison after filling the dems: {dem1_Name} vs {dem2_Name}",ax_x_units ='Elevation (m)')

    ##______ Slope statistics: Compute mean, std, mode, max and min. Compare slope histograms."
        # Compute Slope and Slope stats
    dem_1_Slope = WbT.computeSlope(dem_1_Filled)
    dem_1_SlopStats = computeRaterStats(dem_1_Slope)
    dem_2_Slope = WbT.computeSlope(dem_2_Filled)
    dem_2_SlopeStats  = computeRaterStats(dem_2_Slope)

    # garbageList.append(dem_1_Slope)   ## Uncomment to delete at the end
    # garbageList.append(dem_2_Slope)   ## Uncomment to delete at the end
 
        # Log Slope Stats.
    update_logs({f"{dem1_Name} slope stat ": dem_1_SlopStats})
    update_logs({f"{dem2_Name} slope stat  ": dem_2_SlopeStats})
        # plot elevation histogram
    print("### >>>> Preparing plot......")
    plotRasterHistComparison(dem_1_Slope,dem_2_Slope,title = f'Slope comparison: {dem1_Name} vs {dem2_Name}',bins=[0,1,2,4,6,8,10,15,30,45], ax_x_units= 'Slope (%)')
    
    ### Flow routine: Flow accumulation, d8_pointer, stream network raster, stream network vectors:  
        ##______dem_1 Flow routine.
            # Compute Flow accumulation and Flow accumulation stat on filled cdem.
            # Flow Accumulation statistics: Compute mean, std, mode, max and min. Compare slope histograms."
    dem_1_FAcc = WbT.d8_flow_accumulation(dem_1_Filled, valueType="catchment area")
    dem_1_FAcc_Stats = computeRaterStats(dem_1_FAcc)
    # garbageList.append(dem_1_FAcc)   ## Uncomment to delete at the end

            # River net for 5th and 3rd ostrahler orders. 
    river5th_cdemName = addSubstringToName(dem_1,'_river5thOrder')
    WbT.extractStreamNetwork(dem_1_FAcc,river5th_cdemName,strahOrdThreshold_5th)
    
    river3rd_dem_1_Name = addSubstringToName(dem_1,'_river3rdOrder')
    WbT.extractStreamNetwork(dem_1_FAcc,river3rd_dem_1_Name,strahOrdThreshold_3rd)
    update_logs({f"Flow accumulation stats from {dem1_Name}: ": dem_1_FAcc_Stats})  

    garbageList.append(river3rd_dem_1_Name)
             
            #_ River network vector computed from the 3rd Strahler order river network.
    # Compute Flow Direction with d8FPointerRasterCalculation()
    d8Pionter_dem_1 = WbT.d8FPointerRasterCalculation(dem_1_Filled)    # This is the flow direction map
    river3rd_dem_1_shape = WbT.rasterStreamToVector(river3rd_dem_1_Name,d8Pionter_dem_1)
    
         ##______ dem_2 Flow routine.
            # Compute Flow accumulation and Flow accumulation stats on filled dem_2.
            # Flow Accumulation statistics: Compute mean, std, mode, max and min. Compare slope histograms."
    FAcc_dem_2 = WbT.d8_flow_accumulation(dem_2_Filled, valueType="catchment area")  # 
    FAcc_dem_2_Stats = computeRaterStats(FAcc_dem_2)
    # garbageList.append(FAcc_dem_2)   ## Uncomment to delete at the end
    
    river5th_dem_2_Name = addSubstringToName(dem_2,'_river5thOrder')
    WbT.extractStreamNetwork(FAcc_dem_2,river5th_dem_2_Name,strahOrdThreshold_5th)
    
    river3rd_dem_2_Name = addSubstringToName(dem_2,'_river3rdOrder')
    WbT.extractStreamNetwork(FAcc_dem_2,river3rd_dem_2_Name,strahOrdThreshold_3rd)
    update_logs({f"Flow accumulation stats from {dem2_Name}: ": FAcc_dem_2_Stats})

    garbageList.append(FAcc_dem_2)
    garbageList.append(river3rd_dem_2_Name)

            #_ River network vector computed from the 3rd Strahler order river network.
    # Compute Flow Direction with d8FPointerRasterCalculation()
    d8Pionter_dem_2 = WbT.d8FPointerRasterCalculation(dem_2_Filled)
    river3rd_dem_2_shape = WbT.rasterStreamToVector(river3rd_dem_2_Name, d8Pionter_dem_2)
    plt.show()
    # Print a layOut with both 3rd order river networks vectors. 
    QT.overlap_vectors(river3rd_dem_1_shape,river3rd_dem_2_shape,layOutPath)   

    if emptyGarbage is True:
        for f in garbageList:
            print(f"READY to remove : {f}")
            os.remove(f)
    
#######################
### Rasterio Tools  ###
#######################

def readRasterWithRasterio(rasterPath:os.path) -> tuple[np.array, dict]:
    '''
    Read a raster with Rasterio.
    return:
     Raster data as np.array
     Raster.profile: dictionary with all raster information
    '''
    inRaster = rio.open(rasterPath, mode="r")
    profile = inRaster.profile
    rasterData = inRaster.read()
    # print(f"raster data shape in ReadRaster : {rasterData.shape}")
    return rasterData, profile

def createRaster(savePath:os.path, data:np.array, profile, noData:int = None):
    '''
    Parameter: 
    @savePath: Most contain the file name ex.: *name.tif.
    @data: np.array with shape (bands,H,W)
    '''
    B,H,W = data.shape[-3],data.shape[-2],data.shape[-1] 
    # print(f"C : {B}, H : {H} , W : {W} ")
    profile.update(dtype = rio.uint16, nodata = noData, blockysize = profile['blockysize'])
    with rio.open(
        savePath,
        mode="w",
        #out_shape=(B, H ,W),
        **profile
        ) as new_dataset:
            # print(f"New Dataset.Profile: ->> {new_dataset.profile}")
            new_dataset.write(data)
            print("Created new raster>>>")
    return savePath
   
def plotHistogram(raster, CustomTitle:str = None, bins: int=100, bandNumber: int = 1):
    if CustomTitle is not None:
        title = CustomTitle
    else:
        title = f"Histogram of band : {bandNumber}"    
    data,_ = readRasterWithRasterio(raster)
    show_hist(source=data, bins=bins, title= title, 
          histtype='stepfilled', alpha=0.5)
    return True

def readRasteReplacingNoDataWithNan(rasterPath:os.path,extraNoDataVal: float = None)-> np.array:
    rasterData,profil = readRasterWithRasterio(rasterPath)
    NOData = profil['nodata']
    rasterNoDataAsNan = np.where(((rasterData == NOData)|(rasterData == extraNoDataVal)), np.nan, rasterData) 
    return rasterNoDataAsNan

def computeRaterStats(rasterPath:os.path)-> dict:
    '''
    Read a reaste and return: 
    @Return
    @rasMin: Raster min.
    @rasMax: Raster max.
    @rasMean: Rater mean.
    @rasMode: Raster mode.
    @rasSTD: Raster standard deviation.
    @rasNoNaNCont: Raster count of all valid pixels <NOT NoData>. 
    '''
    rasNoDataAsNan = readRasteReplacingNoDataWithNan(rasterPath)
    rasMin = np.nanmin(rasNoDataAsNan)
    rasMax = np.nanmax(rasNoDataAsNan)
    rasMean = np.nanmean(rasNoDataAsNan)
    rasSTD = np.nanstd(rasNoDataAsNan)
    rasNoNaNCont = np.count_nonzero(rasNoDataAsNan != np.nan)
    # Compute mode
    values,counts = np.unique(rasNoDataAsNan, return_counts=True)
    # remouve Nan index from values and counts.
    counts = np.delete(counts,np.argwhere(np.isnan(values))[0])
    values = np.delete(values,np.argwhere(np.isnan(values))[0])
    index = np.argmax(counts)
    rasMode = values[index]
    report = {'Minim':rasMin,'Max':rasMax, 'Mean':rasMean , 'Mode':rasMode , 'STD':rasSTD, 'Valids Count':rasNoNaNCont}
    return report 

def computeRasterValuePercent(rasterPath, value:int=1)-> float:
    '''
    Compute the percent of pixels of value <value: default =1> in a raster. 
    @rasterPath: Path to the raster to be analyzed.
    @value: Value to verify percent in raster. Default = 1. 
    @return: The computed percent of <value> within the nonNoData values in the input raster.  
    '''
    rasDataNan = readRasteReplacingNoDataWithNan(rasterPath)
    rasNoNaNCont = np.count_nonzero(rasDataNan != np.nan)
    valuCont = np.count_nonzero(rasDataNan == value)
    return (valuCont/rasNoNaNCont)*100

def replace_negative_values(raster_path, fillWith:float = np.nan):
    '''
    This function takes a path to a raster image as input and returns a new raster image with no negative values.
    The function reads the input raster using rasterio, replaces all negative values with <fillWith: default = 0>, and writes the updated data to a new raster file.
    @illWith: float : defaul=0.0  ## You can adapt this value to your needs. 
    '''
    # Open the file with rasterio
    with rio.open(raster_path) as src:
        # Read the raster data
        data = src.read()
        # Replace negative values with 0
        data[data < 0] = fillWith
        # Create a new raster file with the updated data
        new_raster_path = addSubstringToName(raster_path,"_")
        with rio.open(
            new_raster_path,
            "w",
            driver="GTiff",
            width=src.width,
            height=src.height,
            count=1,
            dtype=data.dtype,
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            dst.write(data)
    return new_raster_path


#########################
####   WhiteBoxTools  ###
#########################

## LocalPaths and global variables: to be adapted to your needs ##
currentDirectory = os.getcwd()
wbt = WhiteboxTools()  ## Need to create an instanace on WhiteBoxTools to call the functions.
wbt.set_working_dir(currentDirectory)
print(f"Current dir  {currentDirectory}")
wbt.set_verbose_mode(True)
wbt.set_compress_rasters(True) # compress the rasters map. Just ones in the code is needed

    ## Pretraitment #
class WbT_dtmTransformer():
    '''
     This class contain some functions to generate geomorphological and hydrological features from DEM.
    Functions are based on WhiteBoxTools and Rasterio libraries. For optimal functionality DTM’s most be high resolution, ideally Lidar derived  1m or < 2m. 
    '''
    def __init__(self, workingDir: None) -> None:
        
        
        if workingDir is not None: # Creates output dir if it does not already exist 
            wbt.set_working_dir(workingDir)
            self.workingDir = wbt.get_working_dir()
            print(f"White Box Tools working dir: {self.workingDir}")

    def fixNoDataAndfillDTM(self, inDTMName, eraseIntermediateRasters = True)-> os.path:
        '''
        Ref:   https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html#filldepressions
        To ensure the quality of this process, this method execute several steep in sequence, following the Whitebox’s authors recommendation (For mor info see the above reference).
        Steps:
        1-	Correct no data values to be accepted for all operation. 
        2-	Fill gaps of no data.
        3-	Fill depressions.
        4-	Remove intermediary results to save storage space (Optionally you can keep it. See @Arguments).  
        @Argument: 
        -inDTMName: Input DTM name
        -eraseIntermediateRasters(default = True): Erase intermediate results to save storage space. 
        @Return: True if all process happened successfully, ERROR messages otherwise. 
        @OUTPUT: DTM <filled_ inDTMName> Corrected DTM with wang_and_liu method. 
        '''
        # dtmNoDataValueSetted = addSubstringToName(inDTMName,'_NoDataOK')
        # wbt.set_nodata_value(
        #     inDTMName, 
        #     dtmNoDataValueSetted, 
        #     back_value=0.0, 
        #     callback=default_callback
        #     )
        dtmMissingDataFilled = addSubstringToName(inDTMName,'_')
        wbt.fill_missing_data(
            inDTMName, 
            dtmMissingDataFilled, 
            filter=11, 
            weight=2.0, 
            no_edges=True, 
            callback=default_callback
            )
        output = addSubstringToName(inDTMName,"_fill")  # The DEM is filled with WangAndLiu correction method. 
        wbt.fill_depressions_wang_and_liu(
            dtmMissingDataFilled, 
            output, 
            fix_flats=True, 
            flat_increment=None, 
            callback=default_callback
            )
        if eraseIntermediateRasters:
            try:
                # os.remove(os.path.join(wbt.work_dir,dtmNoDataValueSetted))
                os.remove(os.path.join(wbt.work_dir,dtmMissingDataFilled))
            except OSError as error:
                print("There was an error removing intermediate results : \n {error}")
              
        return output

    def d8FPointerRasterCalculation(self,inFilledDTMName):
        '''
        @argument:
         @inFilledDTMName: DTM without spurious point ar depression.  
        @UOTPUT: D8_pioter: Raster tu use as input for flow direction and flow accumulation calculations. 
        '''
        output = addSubstringToName(inFilledDTMName,"_d8Pointer")
        wbt.d8_pointer(
            inFilledDTMName, 
            output, 
            esri_pntr=False, 
            callback=default_callback
            )
        return output
    
    def d8_flow_accumulation(self, inFilledDTMName, valueType:str = 'cells'):
        '''
        @valueType: Type of contributing area calculation. 
            @valueType Options: one of  (default), 'catchment area', and 'specific contributing area'.
        '''
        d8FAccOutputName = addSubstringToName(inFilledDTMName,"_d8FAcc" ) 
        wbt.d8_flow_accumulation(
            inFilledDTMName, 
            d8FAccOutputName, 
            out_type = valueType, 
            log=False, 
            clip=False, 
            pntr=False, 
            esri_pntr=False, 
            callback=default_callback
            ) 
        return d8FAccOutputName
    
    def jensePourPoint(self,inOutlest,d8FAccOutputName):
        jensenOutput = "correctedSnapPoints.shp"
        wbt.jenson_snap_pour_points(
            inOutlest, 
            d8FAccOutputName, 
            jensenOutput, 
            snap_dist = 15.0, 
            callback=default_callback
            )
        print("jensePourPoint Done")

    def watershedConputing(self,d8Pointer, jensenOutput):  
        output = addSubstringToName(d8Pointer, "_watersheds")
        wbt.watershed(
            d8Pointer, 
            jensenOutput, 
            output, 
            esri_pntr=False, 
            callback=default_callback
        )
        print("watershedConputing Done")

    def DInfFlowCalculation(self, inD8Pointer, log = False):
        ''' 
        Compute DInfinity flow accumulation algorithm.
        Ref: https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html#dinfflowaccumulation  
        We keep the DEFAULT SETTING  from source, which compute "Specific Contributing Area". 
        See ref for the description of more output’s options. 
        @Argument: 
            @inD8Pointer: D8-Pointer raster
            @log (Boolean): Apply Log-transformation on the output raster
        @Output: 
            DInfFlowAcculation map. 
        '''
        output = addSubstringToName(inD8Pointer,"_dInfFAcc")
        wbt.d_inf_flow_accumulation(
            inD8Pointer, 
            output, 
            out_type="Specific Contributing Area", 
            threshold=None, 
            log=log, 
            clip=False, 
            pntr=True, 
            callback=default_callback
            )
        return output

    def StrahlerOrder(self, d8_pntr, streams, output):
        wbt.strahler_stream_order(
            d8_pntr, 
            streams, 
            output, 
            esri_pntr=False, 
            zero_background=False, 
            callback=default_callback
        )
        return output
    
    def extractStreamNetwork(self, FAcc, output, threshold):
        '''
        @FlowAccumulation: Flow accumulation raster.
        @output: Output file path.
        @Threshold: The threshol to determine whethed a cell is starting a river or not. See ref:
            https://www.whiteboxgeo.com/manual/wbt_book/available_tools/stream_network_analysis.html#ExtractStreams
        '''
        wbt.extract_streams(
            FAcc, 
            output, 
            threshold, 
            zero_background=False, 
            callback=default_callback
        )
        return output
    
    def rasterStreamToVector(self, streams, d8_pointer, outVector:str= None):
        '''
        @streams: Stream network in raster format
        @d8_pointer: d8_pointer computed with WbT.d8FPointerRasterCalculation
        @uotVector (Optional): output vector name. If <None>, it'll be created internally. 
        '''
        if outVector != None: output = outVector
        else: output= replaceExtention(streams,".shp")
        print(f"Sheck-in on resterStreamTovector output name: {output}")
        wbt.raster_streams_to_vector(
            streams, 
            d8_pointer, 
            output, 
            esri_pntr=False, 
            callback=default_callback
        )
        return output
    
    def computeSlope(self,inDTMName):
        outSlope = addSubstringToName(inDTMName,'_Slope')
        wbt.slope(inDTMName,
                outSlope, 
                zfactor=None, 
                units="degrees", 
                callback=default_callback
                )
        return outSlope
    
    def computeAspect(self,inDTMName):
        outAspect = addSubstringToName(inDTMName,'_aspect')
        wbt.aspect(inDTMName, 
                outAspect, 
                zfactor=None, 
                callback=default_callback
                )
        return outAspect

    def computeRasterHistogram(self,inRaster):
        '''
        For details see Whitebox Tools references at:
        https://www.whiteboxgeo.com/manual/wbt_book/available_tools/mathand_stats_tools.html?#RasterHistogram
        @return: An *.html file with the computed histogram. The file is autoloader. 
        '''
        output = addSubstringToName(inRaster,'_histogram')
        output = replaceExtention(output, '.html')
        wbt.raster_histogram(
            inRaster, 
            output, 
            callback=default_callback
            )   
        return output

    def rasterCalculator(self, output, statement:str)-> os.path:
        '''
        For details see Whitebox Tools references at:
        https://www.whiteboxgeo.com/manual/wbt_book/available_tools/mathand_stats_tools.html#RasterCalculator
        
        @statement : string of desired operation. Raster must be quoted inside the statement str. ex "'raster1.tif' - 'rater2.tif'"
        '''
        wbt.raster_calculator(
            output, 
            statement, 
            callback=default_callback
            )
        return output

    def get_WorkingDir(self):
        return wbt.get_working_dir()
    
    def set_WorkingDir(self,NewWDir):
        wbt.set_working_dir(NewWDir)

# Helpers
def setWBTWorkingDir(workingDir):
    wbt.set_working_dir(workingDir)
