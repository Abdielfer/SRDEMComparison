import os, ntpath
import glob
import pathlib
import shutil
from time import strftime
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import rasterio as rio
from rasterio.plot import show_hist
from datetime import datetime
from whitebox.whitebox_tools import WhiteboxTools, default_callback
import whitebox_workflows as wbw   
from torchgeo.datasets.utils import download_url
from osgeo import gdal,ogr, osr
from osgeo import gdal_array
from osgeo.gdalconst import *
import pcraster as pcr
from pcraster import *
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
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
def ensureDirectory(pathToCheck:os.path)->os.path:
    if not os.path.isdir(pathToCheck): 
        os.mkdir(pathToCheck)
        print(f"Confirmed directory at: {pathToCheck} ")
    return pathToCheck

def relocateFile(inputFilePath, outputFilePath):
    '''
    NOTE: @outputFilePath must contain the complete filename
    Sintax:
     @shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
    '''
    shutil.move(inputFilePath, outputFilePath)
    return True

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

def plotHistComparison(DEM1,DEM2,title:str='', bins:int = 50):
    # Reding raster. 
    dataReCHaped = np.reshape(readRasterAsArry(DEM1),(1,-1))
    dataReCHaped2 = np.reshape(readRasterAsArry(DEM2),(1,-1))
    # Prepare plot
    fig, ax = plt.subplots(1,sharey=True, tight_layout=True)
    ax.hist(dataReCHaped[0],bins,histtype='step',label=['cdem 16m'])
    ax.hist(dataReCHaped2[0],bins,histtype='step',label=['srdem 8m'])
    ax.legend(prop={'size': 10})
    ax.set_title(title) 
    fig.tight_layout()
    # plt.show()
    
def reporSResDEMComparison(cfg: DictConfig):
    ''', logName:str
    The goal of this function is to create a report of comparison, between cdem and the DEM enhanced with Super Resolution algorithms. 
    
    Data:
    cdem: Canadian Digital Elevation Model at 16m resolution.
    srdem: Super Resolution Algorithm Output at 8m resolution. 

    Goals: Report some geomorphological measurements to evaluate the impact of super-Resolution algorithms in the cdem. 

        The similarity between the source cdem and srdem will be evaluated through statistics and products derived from the DEM. Since both dems are not in the same resolution, the comparison is done in terms of percentage and visual inspection. 

        Histogram interpretaion: Since resolution goes from 16m to 8m, the srdem has 4X the number of pixels of cdem. One should consider this difference when looking at the histogrmas. 

        
    Measurements of comparison (description): 

        - Elevation statistics' summary comparison:
                Compute Min, Man, Mean, STD, Mode and the NoNaNCont. Compare the histograms. 
        
        - Filled area comparison: 
                Fill the dems depressions and pits with WhangAndLiu algorithms. Compute the differnet between the original dem <ex. cdem> and the filled dem <ex. cdem_filled>. Compute the percentage of transformed cells on each dem raster(cdem_16m and srdem_8m). Helps to valuate the inpact of super resolution algorith in the input dem.         

        - Slope statistics' summary:
                Compute Min, Man, Mean, STD, Mode and the NoNaNCont. Compare the histograms. 

        - Flow Accumulation statistics' summary:
                Compute Min, Man, Mean, STD, Mode and the NoNaNCont. Compare the histograms.

        - River network visual comparison:
                Compute strahler order (up to 5th order) and main streams (up to 3rd order). For this, thresholds are setting for each level like 25% of max flow accumulation for 5th order and 
                10% of max flow accumulation for 3rd order. The percent is used for thresholding, based on the fact that every basin is different in shape, size and relieve. 
                Create maps (Vector) with overlaps of both networks for visual inspection. 
            --??? Can we compute IoU to evaluate the river net similarity? 
    '''
    ## Inputs
    in_cdem = cfg['cdemPath']
    in_sr_dem = cfg['SRDEMPath']
    strahOrdThreshold_5th = 30000
    strahOrdThreshold_3rd = 10000
    garbageList = []

    ## Replace negative values. 
    cdem = replace_negative_values(in_cdem)
    sr_dem = replace_negative_values(in_sr_dem)
    
    garbageList.append(cdem)
    garbageList.append(sr_dem)

    ## Inicialize WhiteBoxTools working directory
    if  cfg['WbTools_work_dir'] == 'None':
        perentPath,_,_ = get_parenPath_name_ext(cdem)
        WbTools_wDir = perentPath
    else: 
        WbTools_wDir = cfg['WbTools_work_dir']    
    WbT = WbT_dtmTransformer(WbTools_wDir) 
    
    ## Prepare report loging
    logging.info({"WBTools working dir ": WbT.get_WorkingDir()})
    
    ##______ Elevations tatistics before filling : Compute mean, std, mode, max and min. Compare elevation histograms."
    cdemElevStat = computeRaterStats(cdem)
    srdemElevStats = computeRaterStats(sr_dem)
        # Log Elevation Stats.
    update_logs({"cdem elevation stats befor filling: ": cdemElevStat})
    update_logs({"sr_dem elevation stats befor filling: ": srdemElevStats})
        # plot elevation histogram
    plotHistComparison(cdem,sr_dem,title='Elevation comparison: cdem_16m vs srdem_8m')
    
    ##______ Filled area comparison: Compute mean, std, mode, max and min. Compare the histograms."
        #____Fill the DEMs with WhangAndLiu algorithms from WhiteBoxTools
    cdem_Filled = replace_negative_values(WbT.fixNoDataAndfillDTM(cdem))
    sr_dem_Filled = replace_negative_values(WbT.fixNoDataAndfillDTM(sr_dem))

    garbageList.append(cdem_Filled) 
    garbageList.append(sr_dem_Filled) 

        #___ Compute and log percent of transformed areas. 
            ########  cdem
    cdem_statement = str("'"+cdem_Filled+"'"+'-'+"'"+cdem+"' > 0.05") # Remouve some noice because of aproximations with -0.05
    cdem_transformations = addSubstringToName(in_cdem,'_TransformedArea')
    cdem_Transformations_binary = WbT.rasterCalculator(cdem_transformations,cdem_statement)
    cdem_Transformation_percent = computeRasterValuePercent(cdem_Transformations_binary)
    update_logs({"Depretions an pit percent in cdem ": cdem_Transformation_percent})

    garbageList.append(cdem_Transformations_binary)

            ########  sr_cdem
    sr_cdem_statement = str("'"+sr_dem_Filled+"'"+' - '+"'"+sr_dem+"' > 0.05") # Remouve some noice because of aproximations with -0.05
    sr_cdem_transformations = addSubstringToName(in_sr_dem,'_TransformedArea')
    sr_cdem_Transformations_binary = WbT.rasterCalculator(sr_cdem_transformations,sr_cdem_statement)
    sr_cdem_Transformation_percent = computeRasterValuePercent(sr_cdem_Transformations_binary)
    update_logs({"Depretions an pit percent in sr_cdem ": sr_cdem_Transformation_percent})

    garbageList.append(sr_cdem_Transformations_binary)

   ##______ Elevations statistics AFTER filling : Compute mean, std, mode, max and min. Compare elevation histograms."
    cdemFilledElevStat = computeRaterStats(cdem_Filled)
    srdemFilledElevStats = computeRaterStats(sr_dem_Filled)
        # Log Elevation Stats.
    update_logs({"cdem_Filled elevation stats  ": cdemFilledElevStat})
    update_logs({"sr_dem_Filled elevation stats  ": srdemFilledElevStats})
        # plot elevation histogram of filled dems.
    plotHistComparison(cdem_Filled,sr_dem_Filled,title='Elevation comparison after filling the dems: cdem_16m vs srdem_8m')

    ##______ Slope statistics: Compute mean, std, mode, max and min. Compare slope histograms."
        # Compute Slope and Slope stats
    cdemSlope = WbT.computeSlope(cdem_Filled)
    cdemSlopStats = computeRaterStats(cdemSlope)
    sr_demSlope = WbT.computeSlope(sr_dem_Filled)
    sr_demSlopeStats  = computeRaterStats(sr_demSlope)
    
    garbageList.append(cdemSlope)
    garbageList.append(sr_demSlope)

        # Log Slope Stats.
    update_logs({"cdem slope stat ": cdemSlopStats})
    update_logs({"sr_dem slope stat  ": sr_demSlopeStats})
        # plot elevation histogram
    print("### >>>> Preparing plot......")
    plotHistComparison(cdemSlope,sr_demSlope,title='Slope comparison: cdem_16m vs srdem_8m')
    
    ### Flow routine: Flow accumulation, d8_pointer, stream network raster, stream network vector:  
    
        ##______ cdem Flow routine.
            # Compute Flow accumulation and Flow accumulation's stats on Filled cdem.
            # Flow Accumulation statistics: Compute mean, std, mode, max and min. Compare slope histograms."
    FAcc_cdem = WbT.d8_flow_accumulation(cdem_Filled, valueType="catchment area")
    FAcc_cdem_Stats = computeRaterStats(FAcc_cdem)
    
    garbageList.append(FAcc_cdem)

            # River net for 5th and 3dr ostrahler orders. 
    
    river5th_cdemName = addSubstringToName(in_cdem,'_river5thOrder')
    river5th_cdem = WbT.extractStreamNetwork(FAcc_cdem,river5th_cdemName,strahOrdThreshold_5th)
    
    river3rd_cdemName = addSubstringToName(in_cdem,'_river3rdOrder')
    WbT.extractStreamNetwork(FAcc_cdem,river3rd_cdemName,strahOrdThreshold_3rd)
    update_logs({"Flow accumulation stats from cdem: ": FAcc_cdem_Stats})  

    garbageList.append(river5th_cdem)
             
            #_ River network vector computed from the 5th Strahler order river network.
    d8Pionter_cdem = WbT.d8FPointerRasterCalculation(cdem_Filled)
    WbT.rasterStreamToVector(river5th_cdem,d8Pionter_cdem)
    
    garbageList.append(d8Pionter_cdem)

         ##______ sr_cdem Flow routine.
            # Compute Flow accumulation and Flow accumulation's stats on Filled sr_cdem.
            # Flow Accumulation statistics: Compute mean, std, mode, max and min. Compare slope histograms."
    FAcc_sr_cdem = WbT.d8_flow_accumulation(sr_dem_Filled, valueType="catchment area")  # 
    FAcc_sr_cdem_Stats = computeRaterStats(FAcc_sr_cdem)

    river5th_sr_cdemName = addSubstringToName(in_sr_dem,'_river5thOrder')
    river5th_sr_cdem = WbT.extractStreamNetwork(FAcc_sr_cdem,river5th_sr_cdemName,strahOrdThreshold_5th)
    
    river3rd_sr_cdemName = addSubstringToName(in_sr_dem,'_river3rdOrder')
    WbT.extractStreamNetwork(FAcc_sr_cdem,river3rd_sr_cdemName,strahOrdThreshold_3rd)
    update_logs({"Flow accumulation stats from sr_cdem: ": FAcc_sr_cdem_Stats})

    garbageList.append(FAcc_sr_cdem)
    garbageList.append(river5th_sr_cdem)

            #_ River network vector.
    d8Pionter_sr_dem = WbT.d8FPointerRasterCalculation(sr_dem_Filled)
    WbT.rasterStreamToVector(river5th_sr_cdem, d8Pionter_sr_dem)
    
    garbageList.append(d8Pionter_sr_dem)

    print(garbageList)

    plt.show()

#######################
### Rasterio Tools  ###
#######################

def readRasterWithRasterio(rasterPath:os.path) -> tuple[np.array, dict]:
    '''
    Read a raster with Rasterio.
    return:
     Raster data as np.array
     Raster.profile: dictionary with all rater information
    '''
    inRaster = rio.open(rasterPath, mode="r")
    profile = inRaster.profile
    rasterData = inRaster.read()
    # print(f"raster data shape in ReadRaster : {rasterData.shape}")
    return rasterData, profile

def createRaster(savePath:os.path, data:np.array, profile, noData:int = None):
    '''
    parameter: 
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
   
def plotHistogram(raster, CustomTitle:str = None, bins: int=50, bandNumber: int = 1):
    if CustomTitle is not None:
        title = CustomTitle
    else:
        title = f"Histogram of band : {bandNumber}"    
    data,_ = readRasterWithRasterio(raster)
    
    show_hist(source=data, bins=bins, title= title, 
          histtype='stepfilled', alpha=0.5)
    return True

def replaceRastNoDataWithNan(rasterPath:os.path,extraNoDataVal: float = None)-> np.array:
    rasterData,profil = readRasterWithRasterio(rasterPath)
    NOData = profil['nodata']
    rasterDataNan = np.where(((rasterData == NOData)|(rasterData == extraNoDataVal)), np.nan, rasterData) 
    return rasterDataNan

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
    rasDataNan = replaceRastNoDataWithNan(rasterPath)
    rasMin = np.nanmin(rasDataNan)
    rasMax = np.nanmax(rasDataNan)
    rasMean = np.nanmean(rasDataNan)
    rasSTD = np.nanstd(rasDataNan)
    rasNoNaNCont = np.count_nonzero(rasDataNan != np.nan)
    # Compute mode
    vals,counts = np.unique(rasDataNan, return_counts=True)
    index = np.argmax(counts)
    rasMode = vals[index]
    report = {'Minim':rasMin,'Max':rasMax, 'Mean':rasMean , 'Mode':rasMode , 'STD':rasSTD, 'Valids Count':rasNoNaNCont}
    return report 

def computeRasterValuePercent(rasterPath, value:int=1)-> float:
    '''
    Compute the percent of pixels of value <value: default =1> in a raster. 
    @rasterPath: Path to the raster to be analyzed.
    @value: Value to verify percent in raster. Default = 1. 
    @return: The computed percent of <value> within the nonNoData values in the input raster.  
    '''
    rasDataNan = replaceRastNoDataWithNan(rasterPath)
    rasNoNaNCont = np.count_nonzero(rasDataNan != np.nan)
    valuCont = np.count_nonzero(rasDataNan == value)
    return (valuCont/rasNoNaNCont)*100

def replace_negative_values(raster_path, fillWith:float = 0.0):
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
        new_raster_path = addSubstringToName(raster_path,"_NoNegative")
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

    def fixNoDataAndfillDTM(self, inDTMName, eraseIntermediateRasters = False)-> os.path:
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
        dtmNoDataValueSetted = addSubstringToName(inDTMName,'_NoDataOK')
        wbt.set_nodata_value(
            inDTMName, 
            dtmNoDataValueSetted, 
            back_value=0.0, 
            callback=default_callback
            )
        dtmMissingDataFilled = addSubstringToName(inDTMName,'_MissingDataFilled')
        wbt.fill_missing_data(
            dtmNoDataValueSetted, 
            dtmMissingDataFilled, 
            filter=11, 
            weight=2.0, 
            no_edges=True, 
            callback=default_callback
            )
        output = addSubstringToName(inDTMName,"_filled_WangLiu")
        wbt.fill_depressions_wang_and_liu(
            dtmMissingDataFilled, 
            output, 
            fix_flats=True, 
            flat_increment=None, 
            callback=default_callback
            )
        if eraseIntermediateRasters:
            try:
                os.remove(os.path.join(wbt.work_dir,dtmNoDataValueSetted))
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
        d8FAccOutputName = addSubstringToName(inFilledDTMName,"_d8fllowAcc" ) 
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
        @Threshold: The threshol to determine whethed a cell is staring a river or notr. See ref:
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
        else: output= addSubstringToName(streams,"_vect")
        print(f"Sheck-in on resterStreamTovector output name: {output}")
        wbt.raster_streams_to_vector(
            streams, 
            d8_pointer, 
            output, 
            esri_pntr=False, 
            callback=default_callback
        )
        return outVector
    
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
        @return: An *.html file with the computed histogram. The file is autoloaded. 
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
        
        @statement : string of desired opperation. Raster must be cuoted inside the statement str. ex "'raster1.tif' - 'rater2.tif'"
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

def checkTifExtention(fileName):
    if ".tif" not in fileName:
        newFileName = input("enter a valid file name with the '.tif' extention")
        return newFileName
    else:
        return fileName

def downloadTailsToLocalDir(tail_URL_NamesList, localPath):
    '''
    Import the tails in the url <tail_URL_NamesList>, 
        to the local ydirectory defined in <localPath>.
    '''
    confirmedLocalPath = ensureDirectory(localPath)
    for url in tail_URL_NamesList:
        download_url(url, confirmedLocalPath)
    print(f"Tails downloaded to: {confirmedLocalPath}")

def absolute_value(x):
    return x if x >= 0 else -x