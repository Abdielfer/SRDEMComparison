import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import util as U

@hydra.main(version_base=None, config_path=f"hydraConfig", config_name="mainConfig")
def main(cfg: DictConfig):
    U.reporSResDEMComparison(cfg)
    
    
if __name__ == "__main__":
    with U.timeit():
        main()  
