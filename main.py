import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import hydra 
from omegaconf import DictConfig
import util as U
import logging


@hydra.main(version_base=None, config_path=f"hydraConfig", config_name="mainConfig")
def main(cfg: DictConfig):
    # U.reportSResDEMComparison(cfg)

    U.reportSResDEMComparisonSimplified(cfg)
    
   

if __name__ == "__main__":
    with U.timeit():
        main()  
