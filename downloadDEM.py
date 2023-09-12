import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import hydra 
from omegaconf import DictConfig
import util as U


@hydra.main(version_base=None, config_path=f"hydraConfig", config_name="mainConfig")
def main(cfg: DictConfig):

    downloader = U.generalRasterTools(cfg['wporking_dir'])
    csvName = cfg['csv_demList']
    outputResolution = cfg['out_resolution']
    csvColumn = cfg['csvColumn']
    downloader.mosaikAndResamplingFromCSV(csvName, outputResolution, csvColumn)


if __name__ == "__main__":
    with U.timeit():
        main()  
