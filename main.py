import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import hydra 
from omegaconf import DictConfig
import util as U
import logging
import qgisTools as QT


@hydra.main(version_base=None, config_path=f"hydraConfig", config_name="mainConfig")
def main(cfg: DictConfig):
    # U.reportSResDEMComparison(cfg)
    # dem_1 = cfg['dem_1']
    # dem_2 = cfg['dem_2']
    # _,dem1_Name,_ = U.get_parenPath_name_ext(dem_1) 
    # _,dem2_Name,_ = U.get_parenPath_name_ext(dem_2) 
    # U.plotRasterPDFComparison(dem_1,dem_2,title = f'DEMs correlation {dem1_Name} vs {dem2_Name}',show=True, globalMax=45)
    layOutPath = cfg['layOutOutputPath']
    river3rd_dem_1_shape = r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\Analyses1\cdsm_clip_3\cdsm16mvs8mClip3\cdsm-canada-dem-clip-3979-3_river3rdOrder.shp'
    river3rd_dem_2_shape = r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\Analyses1\cdsm_clip_3\cdsm16mvs8mClip3\SRDEM_x2_256_to_512_cdsm_max_norm_cdsm-canada-dem-clip-3979-3_river3rdOrder.shp'

    QT.overlap_vectors(river3rd_dem_1_shape,river3rd_dem_2_shape,layOutPath)   



if __name__ == "__main__":
    with U.timeit():
        main()  
