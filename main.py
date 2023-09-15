import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import hydra 
from omegaconf import DictConfig
import util as U
import logging


@hydra.main(version_base=None, config_path=f"hydraConfig", config_name="mainConfig")
def main(cfg: DictConfig):
    U.reportSResDEMComparison(cfg)
    # dem_1 = cfg['dem_1']
    # dem_2 = cfg['dem_2']
    # _,dem1_Name,_ = U.get_parenPath_name_ext(dem_1) 
    # _,dem2_Name,_ = U.get_parenPath_name_ext(dem_2) 
    # sampl = U.plotRasterComparisonScattered(dem_1,dem_2,title = f'DEMs correlation {dem1_Name} vs {dem2_Name}',numOfSamples=5000)
    # logging.info(f"ScatterPlot Dataset: {sampl[0:30,:]}")
    # U.createCSVFromList(cfg['layOutOutputPath'],sampl)


if __name__ == "__main__":
    with U.timeit():
        main()  
